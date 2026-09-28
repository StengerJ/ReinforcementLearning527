import java.util.HashSet;
import java.util.Set;

public class GridWorld implements MDP {
    public static final int ROWS = 4;
    public static final int COLS = 4;

    public static final int UP = 0;
    public static final int DOWN = 1;
    public static final int LEFT = 2;
    public static final int RIGHT = 3;

    private static final String[] ACTIONS = {"up", "down", "left", "right"};
    private static final String[] ARROWS = {"^", "v", "<", ">"};

    private static final String[] NAMES = {
        "s1", "s2", "s3", "G2",
        "G1", "s4", "s5", "s6",
        "s7", "s8", "s9", "s10",
        "S", "s11", "s12", "s13"
    };

    public static final int G2 = 3;
    public static final int G1 = 4;
    public static final int START = 12;

    public static final double COLLISION_REWARD = -1.0;
    public static final double STEP_REWARD = 0.0;

    private final boolean[] terminal = new boolean[ROWS * COLS];
    private final double[] entryReward = new double[ROWS * COLS];
    private final Set<String> walls = new HashSet<>();

    public GridWorld() {
        terminal[G1] = true;
        entryReward[G1] = 1.0;
        terminal[G2] = true;
        entryReward[G2] = 5.0;

        addWall(2, 6);
        addWall(6, 7);
        addWall(8, 9);
        addWall(13, 14);
    }

    private void addWall(int a, int b) {
        walls.add(Math.min(a, b) + "-" + Math.max(a, b));
    }

    private boolean blocked(int a, int b) {
        return walls.contains(Math.min(a, b) + "-" + Math.max(a, b));
    }

    public int numStates() {
        return ROWS * COLS;
    }

    public int numActions() {
        return ACTIONS.length;
    }

    public boolean isTerminal(int state) {
        return terminal[state];
    }

    public String stateName(int state) {
        return NAMES[state];
    }

    public String actionName(int action) {
        return ACTIONS[action];
    }

    public static String arrow(int action) {
        return ARROWS[action];
    }

    public Step step(int state, int action) {
        if (terminal[state]) {
            return new Step(0.0, state, true);
        }
        int row = state / COLS;
        int col = state % COLS;
        if (action == UP) {
            row--;
        } else if (action == DOWN) {
            row++;
        } else if (action == LEFT) {
            col--;
        } else {
            col++;
        }
        if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {
            return new Step(COLLISION_REWARD, state, false);
        }
        int next = row * COLS + col;
        if (blocked(state, next)) {
            return new Step(COLLISION_REWARD, state, false);
        }
        if (terminal[next]) {
            return new Step(entryReward[next], next, true);
        }
        return new Step(STEP_REWARD, next, false);
    }
}
