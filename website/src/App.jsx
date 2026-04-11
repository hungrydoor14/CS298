const width_allowed = 1100
const images = {
  carZeroSum: [
    "car-zero-sum-1.png",
    "car-zero-sum-2.png",
  ],
  carGenSum: [
    "car-gensum-1.png",
    "car-gensum-2.png",
    "car-gensum-3.png"
  ],
  carDQNPlanning: [
    "car-dqn-1.png",
    "car-dqn-2.png",
  ],
  dogNash: [
    "dog-blue.png",
    "dog-red.png",
    "dog-solver-1.png",
    "dog-solver-2.png",
    "dog-solver-3.png"
  ],
  dogLearning8: [
    "dog-learning-8-blue.png",
    "dog-learning-8-red.png",
    "dog-learning-8-1.png",
    "dog-learning-8-2.png",
    "dog-learning-8-3.png"
  ],
  dogLearning16: [
    "dog-learning-16-blue.png",
    "dog-learning-16-red.png",
    "dog-learning-16-1.png",
    "dog-learning-16-2.png",
    "dog-learning-16-3.png",
    "dog-learning-16-4.png",
  ],
}

const blue = "#1a6bc4"
const lightBlue = "#e8f0fc"

export default function App() {
  const projects = [
    {
      title: "Car Game",
      intro: "Two players move around a grid. Player 1 (red) tries to collide with Player 2 (blue). Player 2 tries to escape. We solve this using game theory — finding strategies where neither player can do better by changing their move.",
      subs: [
        {
          name: "Zero-Sum Solver",
          description: "The classic version — what's good for one player is exactly bad for the other. We use tabular Q-iteration with fictitious play to find the Nash equilibrium: the pair of strategies where both players are playing their best possible response to each other. Red chases, blue runs, and the math finds the perfect balance.",
          images: images.carZeroSum,
        },
        {
          name: "General-Sum Solver",
          description: "A more realistic twist — both players now have their own independent goals, not just opposites of each other. Player 1 still wants to collide, but Player 2's reward also depends on where it ends up on the grid. We use iterated best-response to find Nash equilibria in this harder setting.",
          images: images.carGenSum,
        },
        {
          name: "DQN Planning",
          description: "Instead of solving the game exactly (which gets expensive fast), we train a neural network to approximate the Q-values. This version uses planning — the agent has access to the environment model and builds its strategy by simulating outcomes. Minimax policy: the red player picks the move that maximizes its worst-case outcome.",
          images: images.carDQNPlanning,
        }
      ],
    },
    {
      title: "Dog Game",
      intro: "Two players move around a continuous 2D space. Their midpoint is a virtual 'dog.' Each player is trying to steer the dog toward their own house. This is a cooperative-competitive setting — they share the dog but have different goals.",
      subs: [
        {
          name: "Nash Solver",
          description: "Both agents use joint Q-networks — a single network that outputs Q-values for every pair of actions both players could take. At each step, we solve for the Nash equilibrium using iterated best-response. An LRU cache stores solutions for states we've already seen, which is the main trick that makes this tractable.",
          images: images.dogNash,
          analysis: "However, because a solver is needed at every step, training is very slow. The agents learn a decent strategy but it's not as polished as the DQN versions below. My efforts would then focus on the learning approach, which is more scalable and ultimately more interesting since it doesn't assume access to the environment model or a solver.",
          
        },
        {
          name: "Learning (8 directions)",
          description: "Each player gets their own independent Q-network and learns on their own — no joint action matrix, no game theory solver. Despite the simplicity, they still learn to cooperate because their rewards are coupled through the dog's position. Soft target network updates keep training stable.",
          images: images.dogLearning8,
          analysis: "With 8 directions the agents learn a coarse policy. Movement is visibly blocky and the dog tends to overshoot the target house before correcting.",
        },
        {
          name: "Learning (16 directions)",
          description: "Each player gets their own independent Q-network and learns on their own — no joint action matrix, no game theory solver. Despite the simplicity, they still learn to cooperate because their rewards are coupled through the dog's position. Soft target network updates keep training stable.",
          images: images.dogLearning16,
          analysis: "16 directions gives much smoother trajectories. The agents develop a more refined cooperative strategy, with the dog taking more direct paths to the target.",

        },
      ],
    },
  ]

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: "#fff", minHeight: "100vh" }}>

      {/* Header */}
      <div style={{ background: blue, color: "#fff", padding: "60px 24px 50px" }}>
        <div style={{ maxWidth: width_allowed, margin: "0 auto" }}>
          <div style={{ fontSize: 12, letterSpacing: "0.1em", textTransform: "uppercase", opacity: 0.7, marginBottom: 12 }}>
            CS298 · Juan José Sandoval Atehortua · Nash Equilibrium
          </div>
          <h1 style={{ fontSize: 36, fontWeight: 700, margin: "0 0 16px", lineHeight: 1.2 }}>
            Multi-Agent Deep Q-Networks
          </h1>
          <p style={{ fontSize: 16, opacity: 0.85, lineHeight: 1.7, margin: 0, maxWidth: 560 }}>
            How do you teach two AI agents to play a game against each other — or with each other — when neither knows what the other will do? This project explores that question through two custom environments and four different approaches, from classical game theory to deep reinforcement learning.
          </p>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: width_allowed, margin: "0 auto", padding: "60px 24px 120px" }}>
        {projects.map((project, pi) => (
          <div key={project.title} style={{ marginBottom: 80 }}>

            {/* Project header */}
            <div style={{ borderLeft: `4px solid ${blue}`, paddingLeft: 16, marginBottom: 20 }}>
              <h2 style={{ fontSize: 24, fontWeight: 700, margin: "0 0 8px", color: "#111" }}>
                {project.title}
              </h2>
              <p style={{ fontSize: 15, color: "#555", lineHeight: 1.7, margin: 0 }}>
                {project.intro}
              </p>
            </div>

            {/* Sub-projects */}
            {project.subs.map((sub, si) => (
              <div key={sub.name} style={{ marginBottom: 56 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                  <div style={{ background: lightBlue, color: blue, fontSize: 11, fontWeight: 600, padding: "3px 10px", borderRadius: 4, letterSpacing: "0.05em" }}>
                    {si + 1 < 10 ? `0${si + 1}` : si + 1}
                  </div>
                  <h3 style={{ fontSize: 18, fontWeight: 600, margin: 0, color: "#111" }}>
                    {sub.name}
                  </h3>
                </div>
                <p style={{ fontSize: 15, color: "#555", lineHeight: 1.75, margin: "0 0 20px" }}>
                  {sub.description}
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {sub.images.map((src, i) => (
                    <img
                      key={i}
                      src={src}
                      alt={`${sub.name} ${i + 1}`}
                      style={{ width: "100%", borderRadius: 8, display: "block", border: "1px solid #e5e7eb" }}
                    />
                  ))}
                </div>
                {sub.analysis && (
  <p style={{ fontSize: 15, color: "#555", lineHeight: 1.75, margin: "20px 0 0", borderTop: "1px solid #e5e7eb", paddingTop: 16 }}>
    {sub.analysis}
  </p>
)}
              </div>
            ))}

            {pi < projects.length - 1 && (
              <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "0 0 80px" }} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}