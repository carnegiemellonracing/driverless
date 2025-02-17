#include "environment.hpp"
#include <unordered_map>
#include <tuple>
#include <memory>

using Reward = double;

namespace controls {
    // An abstractclass implementing a continuous action type
    struct Action {
        virtual ~Action() = default;
    };
    
    // An abstract struct implementing a continuous observation type
    struct Observation {
        virtual ~Observation() = default;
    };

    class ObservationSpace {  
        public:
            ObservationSpace(
                Observation initialObservation, 
                Observation (*transitionFunction)(Observation, Action),
                bool (*isTerminal)(Observation), 
                bool (*isTruncated)(Observation)
            ) {
                this->currState = initialObservation;
                this->transitionFunction = transitionFunction;
                this->isTerminal = isTerminal;
                this->isTruncated = isTruncated;
            };

            Observation observe() {
                return this->currState;
            };

            std::tuple<Observation, bool, bool> step(Action a) {
                Observation nextState = (this->transitionFunction)(this->currState, a);
                bool isTerminal = (this->isTerminal)(nextState);
                bool isTruncated = (this->isTruncated)(nextState);
                return std::make_tuple(nextState, isTerminal, isTruncated);
            };

            // Resets the ObservationSpace, returning an initial observation
            Observation reset() {
                this->currState = initialState;
                return this->initialState;
            };
            
            // Returns [true] if [Observation o] is terminal
            bool isTerminalObservation(Observation o) {
                return (this->isTerminal)(o);
            };

            // Returns [true] if [Observation o] has been truncated
            // I.e., whether it has gone out of bounds or went over a time limit
            bool isTruncatedObservation(Observation o) {
                return (this->isTruncated)(o);
            };

        private:
            Observation initialState;
            Observation currState;
            Observation (*transitionFunction)(Observation, Action);
            bool (*isTerminal)(Observation);
            bool (*isTruncated)(Observation);
    };


    // A class for rewards functions. 
    // Utilizes function pointers over std::function for performance
    class RewardFunction {
        public:
            RewardFunction(Reward (*rewardFunction)(Observation, Action, Observation)) {
                this->rewardFunction = rewardFunction;
            };
            
            Reward getRewards(Observation s_curr, Action a, Observation s_next) {
                return (this->rewardFunction)(s_curr, a, s_next);
            };

        private:
            // Utilizes function pointers over std::function for performance
            Reward (*rewardFunction)(Observation, Action, Observation);
    };

    class Environment {
        public:
            std::unordered_map<std::string, std::string> info;

            Environment(
                Observation initialState,
                Observation (*transitionFunction)(Observation, Action),
                bool (*isTerminal)(Observation), 
                bool (*isTruncated)(Observation),
                Reward (*rewardFunction)(Observation, Action, Observation)
            ) :
                stateSpace{initialState, transitionFunction, isTerminal, isTruncated},
                rewardFunction{rewardFunction}
            {}; 

            // Steps the observation space and returns a tuple of <s', R, truncatedState, terminalState>
            std::tuple<Observation, Reward, bool, bool> step(Action a) {
                Observation currState = (this->stateSpace).observe();
                auto [nextState, truncatedState, terminalState] = (this->stateSpace).step(a);
                Reward r = getRewards(currState, a, nextState);
                return std::make_tuple(nextState, r, truncatedState, terminalState);
            }; 

            // Invokes the rewardFunction, returing Reward for <s, a, s'> tuple.
            Reward getRewards(Observation s_curr, Action a, Observation s_next) {
                return (this->rewardFunction).getRewards(s_curr, a, s_next);
            };

            std::pair<Observation, std::unordered_map<std::string, std::string>> reset() {
                Observation initialObservation = (this->stateSpace).reset();
                return {initialObservation, info};
            };

            // Environment Samplers
            
            // ===== ObservationSpace setters & getters =====
            void setObservationSpace(ObservationSpace observationSpace) {
                this->stateSpace = observationSpace;
            };

            ObservationSpace getObservationSpace() {
                return this->stateSpace;
            };

            // ===== RewardFunction setters & getters =====
            void setRewardFunction(RewardFunction rewardFunction) {
                this->rewardFunction = rewardFunction;
            };
            RewardFunction getRewardFunction() {
                return this->rewardFunction;
            };

        private:
            ObservationSpace stateSpace;
            RewardFunction rewardFunction;
    };
}