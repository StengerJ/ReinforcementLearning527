import java.util.List;

public class Main {

    static final double GAMMA = 0.9;
    static final double TOL = 1e-6;
    static final double INIT = 0.5;
    static final double[] GAMMAS = {0.20, 0.40, 0.60, 0.65, 0.70, 0.80, 0.90, 0.99};
    static final double TIE = 1e-9;

    public static void main(String[] args) {
        GridWorld env = new GridWorld();

        line();
        System.out.println("HW1: DYNAMIC PROGRAMMING IN A GRIDWORLD");
        System.out.println("gamma = " + GAMMA + "   tolerance = " + TOL + "   initial value = " + INIT);
        line();

        printNames(env);

        part1(env);
        Result eval = part2(env);
        part3(env, eval);
        Result pi = part4(env);
        part5(env, pi);
        part6(env);
    }

    static void part1(GridWorld env) {
        header("PART 1: ENVIRONMENT MODEL - DEMONSTRATION TRANSITIONS");
        show(env, 10, GridWorld.RIGHT);
        show(env, 8, GridWorld.UP);
        show(env, 7, GridWorld.UP);
        show(env, 6, GridWorld.RIGHT);
        show(env, 12, GridWorld.LEFT);
    }

    static Result part2(GridWorld env) {
        header("PART 2: ITERATIVE POLICY EVALUATION (uniform random policy)");
        Result r = DP.evaluate(env, DP.uniformPolicy(env), GAMMA, TOL, INIT);
        printValues(env, r.history.get(0), "V0");
        printValues(env, r.history.get(1), "V1");
        printValues(env, r.history.get(2), "V2");
        printValues(env, r.values, "Converged V^pi0");
        System.out.println();
        System.out.println("Sweeps to convergence: " + r.sweeps);
        return r;
    }

    static void part3(GridWorld env, Result eval) {
        header("PART 3: POLICY IMPROVEMENT (greedy with respect to V^pi0)");
        System.out.println("Tie-break rule: lowest action index wins, order = up, down, left, right.");
        printQ(env, eval.values, 6);
        printQ(env, eval.values, 7);
        printQ(env, eval.values, 13);
        printPolicy(env, DP.greedy(env, eval.values, GAMMA), "Improved policy pi1");
    }

    static Result part4(GridWorld env) {
        header("PART 4: POLICY ITERATION");
        Result r = DP.policyIteration(env, GAMMA, TOL, INIT);
        printPolicy(env, r.policy, "Final policy");
        printValues(env, r.values, "Final value function");
        System.out.println();
        System.out.println("Policy-improvement steps: " + r.improvements);
        System.out.println("Total evaluation sweeps:  " + r.sweeps);
        printPath(env, r.policy);
        return r;
    }

    static Result part5(GridWorld env, Result pi) {
        header("PART 5: VALUE ITERATION");
        Result r = DP.valueIteration(env, GAMMA, TOL, INIT);
        printValues(env, r.values, "Final value function");
        printPolicy(env, r.policy, "Greedy policy");
        printPath(env, r.policy);
        double diff = 0.0;
        boolean same = true;
        for (int s = 0; s < env.numStates(); s++) {
            diff = Math.max(diff, Math.abs(r.values[s] - pi.values[s]));
            if (!env.isTerminal(s) && r.policy[s] != pi.policy[s]) {
                same = false;
            }
        }
        System.out.println();
        System.out.println("Max |V_PI - V_VI|      : " + String.format("%.10f", diff));
        System.out.println("Policies identical     : " + same);
        System.out.println("Value-iteration sweeps : " + r.sweeps);
        System.out.println("Policy-iteration sweeps: " + pi.sweeps + " (over " + pi.improvements + " improvement steps)");
        return r;
    }

    static void part6(GridWorld env) {
        header("PART 6: DISCOUNT-FACTOR EXPERIMENT");
        System.out.println(String.format("%-8s %12s %14s %8s %8s %8s",
                "gamma", "V*(S)", "first action", "goal", "steps", "sweeps"));
        for (double g : GAMMAS) {
            Result r = DP.valueIteration(env, g, TOL, INIT);
            List<Integer> path = DP.rollout(env, r.policy, GridWorld.START, 100);
            int last = path.get(path.size() - 1);
            boolean done = env.isTerminal(last);
            System.out.println(String.format("%-8.2f %12.6f %14s %8s %8s %8d",
                    g, r.values[GridWorld.START], env.actionName(r.policy[GridWorld.START]),
                    done ? env.stateName(last) : "none",
                    done ? String.valueOf(path.size() - 1) : "-",
                    r.sweeps));
        }
    }

    static void show(GridWorld env, int s, int a) {
        Step t = env.step(s, a);
        String kind = t.terminal ? "goal entered" : (t.nextState == s ? "collision" : "ordinary move");
        System.out.println(String.format("step(%-4s, %-5s) -> r = %+.1f, next = %-4s, terminal = %-5s   [%s]",
                env.stateName(s), env.actionName(a), t.reward, env.stateName(t.nextState), t.terminal, kind));
    }

    static void printQ(GridWorld env, double[] v, int s) {
        double[] q = DP.qValues(env, v, GAMMA, s);
        double best = q[0];
        for (double x : q) {
            best = Math.max(best, x);
        }
        System.out.println();
        System.out.println("q(" + env.stateName(s) + ", a):");
        StringBuilder ties = new StringBuilder();
        for (int a = 0; a < q.length; a++) {
            boolean isBest = Math.abs(q[a] - best) < TIE;
            System.out.println(String.format("    %-6s %10.6f%s", env.actionName(a), q[a], isBest ? "  <== greedy" : ""));
            if (isBest) {
                ties.append(ties.length() == 0 ? "" : ", ").append(env.actionName(a));
            }
        }
        System.out.println("    greedy set = {" + ties + "}, selected = "
                + env.actionName(DP.greedy(env, v, GAMMA)[s]));
    }

    static void printPath(GridWorld env, int[] policy) {
        List<Integer> path = DP.rollout(env, policy, GridWorld.START, 100);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < path.size(); i++) {
            sb.append(i == 0 ? "" : " -> ").append(env.stateName(path.get(i)));
        }
        int last = path.get(path.size() - 1);
        System.out.println();
        System.out.println("Path from S: " + sb);
        System.out.println("Goal reached: " + (env.isTerminal(last) ? env.stateName(last) : "none")
                + " in " + (path.size() - 1) + " steps");
    }

    static void printNames(GridWorld env) {
        System.out.println();
        System.out.println("State layout (index in parentheses):");
        for (int r = 0; r < GridWorld.ROWS; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < GridWorld.COLS; c++) {
                int s = r * GridWorld.COLS + c;
                sb.append(String.format("%10s", env.stateName(s) + "(" + s + ")"));
            }
            System.out.println(sb);
        }
        System.out.println();
        System.out.println("Walls between: s3|s5, s5|s6, s7|s8, s11|s12");
    }

    static void printValues(GridWorld env, double[] v, String title) {
        System.out.println();
        System.out.println(title + ":");
        for (int r = 0; r < GridWorld.ROWS; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < GridWorld.COLS; c++) {
                sb.append(String.format("%10.4f", v[r * GridWorld.COLS + c]));
            }
            System.out.println(sb);
        }
    }

    static void printPolicy(GridWorld env, int[] policy, String title) {
        System.out.println();
        System.out.println(title + ":");
        for (int r = 0; r < GridWorld.ROWS; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < GridWorld.COLS; c++) {
                int s = r * GridWorld.COLS + c;
                sb.append(String.format("%6s", env.isTerminal(s) ? env.stateName(s) : GridWorld.arrow(policy[s])));
            }
            System.out.println(sb);
        }
    }

    static void header(String text) {
        System.out.println();
        line();
        System.out.println(text);
        line();
    }

    static void line() {
        System.out.println("======================================================================");
    }
}
