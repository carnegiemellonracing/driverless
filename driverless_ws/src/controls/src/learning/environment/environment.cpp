#include "environment.hpp"
#include <unordered_map>
#include <tuple>
#include <memory>

using Reward = double;

namespace controls {
    // An std::shared_ptr<Action> class implementing a continuous action type
    class Action {
        virtual ~Action() = default;
    };

    // An std::shared_ptr<Observation> class implementing a continuous observation type
    class Observation {
        virtual ~Observation() = default;
    };

    class ObservationSpace {   
        public:
            ObservationSpace(
                std::shared_ptr<Observation> initialObservation, 
                std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>),
                bool (*isTerminal)(std::shared_ptr<Observation>), 
                bool (*isTruncated)(std::shared_ptr<Observation>)
            ) {
                this->currState = initialObservation;
                this->transitionFunction = transitionFunction;
                this->isTerminal = isTerminal;
                this->isTruncated = isTruncated;
            };

            std::shared_ptr<Observation> observe() {
                return this->currState;
            };

            std::tuple<std::shared_ptr<Observation>, bool, bool> step(std::shared_ptr<Action> a) {
                std::shared_ptr<Observation> nextState = (this->transitionFunction)(this->currState, a);
                bool isTerminal = (this->isTerminal)(nextState);
                bool isTruncated = (this->isTruncated)(nextState);
                return std::make_tuple(nextState, isTerminal, isTruncated);
            };

            // Resets the ObservationSpace, returning an initial observation
            std::shared_ptr<Observation> reset() {
                this->currState = initialState;
                return this->initialState;
            };
            
            // Returns [true] if [std::shared_ptr<Observation> o] is terminal
            bool isTerminalObservation(std::shared_ptr<Observation> o) {
                return (this->isTerminal)(o);
            };

            // Returns [true] if [std::shared_ptr<Observation> o] has been truncated
            // I.e., whether it has gone out of bounds or went over a time limit
            bool isTruncatedObservation(std::shared_ptr<Observation> o) {
                return (this->isTruncated)(o);
            };

        private:
            std::shared_ptr<Observation> initialState;
            std::shared_ptr<Observation> currState;
            std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>);
            bool (*isTerminal)(std::shared_ptr<Observation>);
            bool (*isTruncated)(std::shared_ptr<Observation>);
    };


    // A class for rewards functions. 
    // Utilizes function pointers over std::function for performance
    class RewardFunction {
        public:
            RewardFunction(Reward (*rewardFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>, std::shared_ptr<Observation>)) {
                this->rewardFunction = rewardFunction;
            };
            
            Reward getRewards(std::shared_ptr<Observation> s_curr, std::shared_ptr<Action> a, std::shared_ptr<Observation> s_next) {
                return (this->rewardFunction)(s_curr, a, s_next);
            };

        private:
            Reward (*rewardFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>, std::shared_ptr<Observation>);
    };

    class Environment {
        public:
            std::unordered_map<std::string, std::string> info;

            Environment(
                std::shared_ptr<Observation> initialState,
                std::shared_ptr<Observation> (*transitionFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>),
                bool (*isTerminal)(std::shared_ptr<Observation>), 
                bool (*isTruncated)(std::shared_ptr<Observation>),
                Reward (*rewardFunction)(std::shared_ptr<Observation>, std::shared_ptr<Action>, std::shared_ptr<Observation>)
            ) :
                stateSpace{initialState, transitionFunction, isTerminal, isTruncated},
                rewardFunction{rewardFunction}
            {}; 

            // Steps the observation space and returns a tuple of <s', R, truncatedState, terminalState>
            std::tuple<std::shared_ptr<Observation>, Reward, bool, bool> step(std::shared_ptr<Action> a) {
                std::shared_ptr<Observation> currState = (this->stateSpace).observe();
                auto [nextState, truncatedState, terminalState] = (this->stateSpace).step(a);
                Reward r = getRewards(currState, a, nextState);
                return std::make_tuple(nextState, r, truncatedState, terminalState);
            }; 

            // Invokes the rewardFunction, returing Reward for <s, a, s'> tuple.
            Reward getRewards(std::shared_ptr<Observation> s_curr, std::shared_ptr<Action> a, std::shared_ptr<Observation> s_next) {
                return (this->rewardFunction).getRewards(s_curr, a, s_next);
            };

            std::pair<std::shared_ptr<Observation>, std::unordered_map<std::string, std::string>> reset() {
                std::shared_ptr<Observation> initialObservation = (this->stateSpace).reset();
                return {initialObservation, info};
            };
            
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