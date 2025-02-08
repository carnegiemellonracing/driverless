#include <tuple>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <memory>

using Reward = double;

namespace controls {
    // A std::shared_ptr<Action> class implementing an abstract action type
    class Action {
        virtual ~Action() = default;
    };

    // A std::shared_ptr<Observation> class implementing an abstract observation type
    class Observation {
        virtual ~Observation() = default;
    };

    class ObservationSpace {   
        public:
            ObservationSpace(
                std::shared_ptr<Observation> initialObservation, 
                std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, Action),
                bool (*isTerminal)(std::shared_ptr<Observation>), 
                bool (*isTruncated)(std::shared_ptr<Observation>)
            );

            std::shared_ptr<Observation> observe() {
                return this->currState;
            };

            std::tuple<std::shared_ptr<Observation>, bool, bool> step(Action a);

            // Resets the ObservationSpace, returning an initial observation
            std::shared_ptr<Observation> reset();
            
            // Returns [true] if [std::shared_ptr<Observation> o] is terminal
            bool isTerminalObservation(std::shared_ptr<Observation> o);

            // Returns [true] if [std::shared_ptr<Observation> o] has been truncated
            bool isTruncatedObservation(std::shared_ptr<Observation> o);

        private:
            std::shared_ptr<Observation> initialState;
            std::shared_ptr<Observation> currState;
            std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, Action);
            bool (*isTerminal)(std::shared_ptr<Observation>);
            bool (*isTruncated)(std::shared_ptr<Observation>);
    };


    // A class for rewards functions. 
    // Utilizes function pointers over std::function for performance
    class RewardFunction {
        public:
            RewardFunction(Reward (*rewardFunction)(std::shared_ptr<Observation>, Action, std::shared_ptr<Observation>));
            Reward getRewards(std::shared_ptr<Observation> s_curr, Action a, std::shared_ptr<Observation> s_next);

        private:
            Reward (*rewardFunction)(std::shared_ptr<Observation>, Action, std::shared_ptr<Observation>);
    };

    class Environment {
        public:
            std::unordered_map<std::string, std::string> info;

            Environment(
                std::shared_ptr<Observation> initialState,
                std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, Action),
                bool (*isTerminal)(std::shared_ptr<Observation>), 
                bool (*isTruncated)(std::shared_ptr<Observation>),
                Reward (*rewardFunction)(std::shared_ptr<Observation>, Action, std::shared_ptr<Observation>)
            ); 
            // ===== Environment Methods =====
            // Steps the ObservationSpace and returns a tuple of <s', R, truncatedState, terminalState>
            std::tuple<std::shared_ptr<Observation>, Reward, bool, bool> step(Action a); 

            // Invokes the rewardFunction, returing Reward for <s, a, s'> tuple.
            Reward getRewards(std::shared_ptr<Observation> s_curr, Action a, std::shared_ptr<Observation> s_next);

            // Resets the ObservationSpace, returning an initial observation and some info parameter
            std::pair<std::shared_ptr<Observation>, std::unordered_map<std::string, std::string>> reset();
            
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