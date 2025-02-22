#include <fstream>
#include <unordered_map>
#include <tuple>
#include <memory>

#ifndef ABSTRACT_RL_ENVIRONMENT
#define ABSTRACT_RL_ENVIRONMENT

using Reward = double;

namespace controls {
    // An abstract Action type
    struct Action {
        virtual ~Action() = default;
    };

    // An abstract Observation type
    struct Observation {
        virtual ~Observation() = default;
    };

    class ObservationSpace {   
        public:
            ObservationSpace();

            ObservationSpace(
                Observation initialObservation, 
                Observation (*transitionFunction)(Observation, Action),
                bool (*isTerminal)(Observation), 
                bool (*isTruncated)(Observation)
            );

            Observation observe() {
                return this->currState;
            };

            std::tuple<Observation, bool, bool> step(Action a);

            // Resets the ObservationSpace, returning an initial observation
            Observation reset();
            
            // Returns [true] if [Observation o] is terminal
            bool isTerminalObservation(Observation o);

            // Returns [true] if [Observation o] has been truncated
            bool isTruncatedObservation(Observation o);

        private:
            Observation initialState;
            Observation currState;
            Observation (*transitionFunction)(Observation, Action);
            bool (*isTerminal)(Observation);
            bool (*isTruncated)(Observation);
    };

    // A class for rewards functions. 
    class RewardFunction {
        public:
            RewardFunction(Reward (*rewardFunction)(Observation, Action, Observation));
            Reward getRewards(Observation s_curr, Action a, Observation s_next);

        private:
            Reward (*rewardFunction)(Observation, Action, Observation);
    };

    class Environment {
        public:
            std::unordered_map<std::string, std::string> info;

            // <c_o, s_o, transition function, terminal state checker, truncated episode checker, reward function>
            Environment(
                Observation initialState,
                Observation (*transitionFunction)(Observation, Action),
                bool (*isTerminal)(Observation), 
                bool (*isTruncated)(Observation),
                Reward (*rewardFunction)(Observation, Action, Observation)
            ); 

            // ===== Environment Methods =====
            // Steps the ObservationSpace and returns a tuple of <s', R, truncatedState, terminalState>
            std::tuple<Observation, Reward, bool, bool> step(Action a); 

            // Invokes the rewardFunction, returing Reward for <s, a, s'> tuple.
            Reward getRewards(Observation s_curr, Action a, Observation s_next);

            // Resets the ObservationSpace, returning an initial observation and some info dictionary
            std::pair<Observation, std::unordered_map<std::string, std::string>> reset();
            
            // ===== ObservationSpace setters & getters =====
            void setObservationSpace(ObservationSpace observationSpace);
            ObservationSpace getObservationSpace();

            // ===== RewardFunction setters & getters =====
            void setRewardFunction(RewardFunction rewardFunction);
            RewardFunction getRewardFunction();

        private:
            ObservationSpace stateSpace;
            RewardFunction rewardFunction;  
    };
}

#endif