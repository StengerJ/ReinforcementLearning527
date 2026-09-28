public interface MDP {
    int numStates();

    int numActions();

    boolean isTerminal(int state);

    Step step(int state, int action);

    String stateName(int state);

    String actionName(int action);
}
