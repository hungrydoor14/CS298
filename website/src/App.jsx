const images = {
  carSolver: [
    "https://placehold.co/600x400?text=Car+Zero-Sum",
    "https://placehold.co/600x400?text=Car+Gen-Sum",
  ],
  carPlanning: [
    "https://placehold.co/600x400?text=Car+Planning+1",
  ],
  dogSolver: [
    "https://placehold.co/600x400?text=Dog+Solver+1",
  ],
  dogLearning: [
    "https://placehold.co/600x400?text=Dog+Learning+1",
  ],
}

export default function App() {
  const projects = [
    {
      title: "Car Game",
      subs: [
        {
          name: "Solver",
          description:
            "Two-player zero-sum grid game. Uses neural planning — samples states and computes Bellman targets directly from the environment model. Minimax policy: Player 1 maximizes the guaranteed value against a worst-case opponent.",
          images: images.carSolver,
        },
        {
          name: "Planning",
          description:
            "Same CarGame environment but trained with a full online DQN loop — epsilon-greedy exploration, experience replay buffer, and a frozen target network. Saddle gap diagnostic tracks convergence to equilibrium.",
          images: images.carPlanning,
        },
      ],
    },
    {
      title: "Dog Game",
      subs: [
        {
          name: "Solver (slow)",
          description:
            "Cooperative continuous-space game where two players jointly steer a dog (their midpoint) toward a house. Uses Nash equilibrium strategies via iterated best-response. Joint Q-networks with an LRU cache to avoid recomputing Nash for seen states.",
          images: images.dogSolver,
        },
        {
          name: "Learning",
          description:
            "Same DogGame but each agent has its own independent Q-network. Agents learn cooperatively via independent Q-learning with soft target updates.",
          images: images.dogLearning,
        },
      ],
    },
  ]

  return (
    <div style={{ maxWidth: 680, margin: "0 auto", padding: "60px 24px 120px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 600, marginBottom: 6 }}>Qiter DQN</h1>
      <p style={{ color: "#888", fontSize: 15, marginBottom: 64 }}>
        Multi-agent deep Q-network experiments.
      </p>

      {projects.map((project) => (
        <div key={project.title} style={{ marginBottom: 80 }}>
          <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 40, borderBottom: "1px solid #eee", paddingBottom: 12 }}>
            {project.title}
          </h2>

          {project.subs.map((sub) => (
            <div key={sub.name} style={{ marginBottom: 60 }}>
              <h3 style={{ fontSize: 16, fontWeight: 500, marginBottom: 12, color: "#444" }}>
                {sub.name}
              </h3>
              <p style={{ fontSize: 14, color: "#666", lineHeight: 1.7, margin: "0 0 20px" }}>
                {sub.description}
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {sub.images.map((src, i) => (
                  <img
                    key={i}
                    src={src}
                    alt={`${sub.name} ${i + 1}`}
                    style={{ width: "100%", borderRadius: 8, display: "block" }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}