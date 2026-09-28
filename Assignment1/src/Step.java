public class Step {
    public final double reward;
    public final int nextState;
    public final boolean terminal;

    public Step(double reward, int nextState, boolean terminal) {
        this.reward = reward;
        this.nextState = nextState;
        this.terminal = terminal;
    }

    public String toString() {
        return "(r=" + reward + ", s'=" + nextState + ", terminal=" + terminal + ")";
    }
}
