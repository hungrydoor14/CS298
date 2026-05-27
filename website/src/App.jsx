import { useEffect, useRef, useState } from "react"

const width_allowed = 1100
const images = {
  carZeroSum: [
    ["car-zero-sum-1.png", 55, 0],
    ["car-zero-sum-2.png", 55, 0, 0.975],
  ],
  carGenSum: [
    ["car-gensum-1.png", 55, 0],
    ["car-gensum-2.png", 55, 0, 0.99],
    ["car-gensum-3.png", 55, 0],
  ],
  carDQNPlanning: [
    ["car-dqn-1.png", 55, 0],
    ["car-dqn-2.png", 60, 5],
  ],
  dogNash: [
    ["dog-blue.png", 5, 5, 0.965],
    ["dog-red.png", 5, 0],
    ["dog-solver-1.png", 0, 0],
    ["dog-solver-2.png", 0, 0, .995],
    ["dog-solver-3.png", 0, 0],
  ],
  dogLearning8: [
    ["dog-learning-8-blue.png", 0, 0],
    ["dog-learning-8-red.png", 0, 0],
    ["dog-learning-8-1.png", 30, 0],
    ["dog-learning-8-2.png", 0, 0, .975],
    ["dog-learning-8-3.png", 0, 0],
  ],
  dogLearning16: [
    ["dog-learning-16-blue.png", 0, 0],
    ["dog-learning-16-red.png", 0, 0, .99],
    ["dog-learning-16-1.png", 0, 0, .99],
    ["dog-learning-16-2.png", 0, 0],
    ["dog-learning-16-3.png", 0, 0],
    ["dog-learning-16-4.png", 0, 0],
  ],
  territoryStandardNoWalls: [
    ["territory-nw-1.png", 65, 0],
    ["territory-nw-2.png", 65, 0],
    ["territory-nw-3.png", 65, 0],
  ],
  territoryStandardWalls: [
    ["territory-w-1.png", 65, 0],
    ["territory-w-2.png", 65, 0],
  ],
  territoryChunkNoWalls: [
    ["territory-nw-chunk-1.png", 65, 0],
  ],
  territoryChunkWalls: [
    ["territory-w-chunk-1.png", 65, 0],
    ["territory-w-chunk-2.png", 65, 0],
  ],
}

const blue = "#1a6bc4"
const lightBlue = "#e8f0fc"
const overview = [
  "The project compares several ways of making two agents learn in the same environment. Some versions solve small Markov games directly with Q-iteration, while others use neural networks to approximate Q-values when the state space becomes too large to enumerate cleanly.",
  "The car experiments are discrete grid games. The exact solvers enumerate joint positions and solve local action matrices, while the DQN version normalizes the four car coordinates and predicts values for all 16 joint actions.",
  "The dog experiments move into continuous 2D control. The state includes both players and both houses, and the dog is computed as the midpoint between the players, so both agents affect the same object even when their goals differ.",
  "The territory experiments use independent DQN agents on a larger grid. Legal action masks, frontier-distance features, wall handling, and chunk features help the agents learn expansion behavior while keeping moves valid.",
]
const limitations = [
  "The exact car solvers are useful for small grids, but full state enumeration becomes expensive as the grid grows.",
  "The dog Nash solver is computationally heavy because training repeatedly estimates strategic responses over joint-action matrices.",
  "The independent dog learners scale better, but they no longer explicitly solve an equilibrium and can learn unstable or coarse coordination.",
  "The territory agents mostly optimize expansion and frontier progress, so adversarial blocking is still only partially represented in the reward.",
]
const futureWork = [
  "Use CNN-based territory policies so agents can reason over the full board layout instead of mostly hand-built local features.",
  "Add stronger blocking and denial rewards based on how much reachable territory the opponent loses after a move.",
  "Compare the DQN variants with actor-critic or policy-gradient methods for the continuous dog game.",
  "Turn the rollout viewers into more interactive browser visualizations so policies can be inspected without rerunning Python scripts.",
]

function ProjectImage({ src, alt, cropBottom, cropTop, scale = 1 }) {
  const imgRef = useRef(null)
  const [visibleHeight, setVisibleHeight] = useState(null)

  useEffect(() => {
    if ((cropBottom <= 0 && cropTop <= 0) || !imgRef.current) return undefined

    const img = imgRef.current
    const updateHeight = () => {
      setVisibleHeight(Math.max(1, img.getBoundingClientRect().height - cropBottom - cropTop))
    }
    const observer = new ResizeObserver(updateHeight)

    updateHeight()
    observer.observe(img)

    return () => observer.disconnect()
  }, [cropBottom, cropTop, src])

  if (cropBottom <= 0 && cropTop <= 0) {
    return (
      <img
        src={src}
        alt={alt}
        style={{ width: `${scale * 100}%`, margin: "0 auto", borderRadius: 8, display: "block", border: "1px solid #e5e7eb" }}
      />
    )
  }

  return (
    <div
      style={{
        width: "100%",
        height: visibleHeight ?? "auto",
        overflow: "hidden",
        borderRadius: 8,
        border: "1px solid #e5e7eb",
      }}
    >
      <img
        ref={imgRef}
        src={src}
        alt={alt}
        onLoad={() => {
          if (imgRef.current) {
            setVisibleHeight(Math.max(1, imgRef.current.getBoundingClientRect().height - cropBottom - cropTop))
          }
        }}
        style={{ width: `${scale * 100}%`, margin: "0 auto", display: "block", transform: `translateY(${-cropTop}px)` }}
      />
    </div>
  )
}

export default function App() {
  const projects = [
    {
      title: "Car Game",
      intro: "Two players move around a grid. The state is both players' positions, and each turn both choose one of four moves: up, down, left, or right. The experiments compare exact Markov-game solvers with a neural planning approximation.",
      info: [
        {
          label: "Environment",
          text: "A discrete grid-world game with two agents moving simultaneously on the same board.",
        },
        {
          label: "State",
          text: "The state tracks both players' grid positions, so each decision depends on the relative distance and possible next moves.",
        },
        {
          label: "Actions",
          text: "Each player chooses a movement action on the grid at every step. The next state is determined by both choices together.",
        },
        {
          label: "Goal",
          text: "The exact reward depends on the variant: the zero-sum and planning versions optimize one shared payoff, while the general-sum version gives each player a separate payoff.",
        },
      ],
      subs: [
        {
          name: "Zero-Sum Solver",
          description: "The classic version uses one payoff matrix, with Player 1 maximizing it and Player 2 minimizing it. The code runs tabular Markov-game Q-iteration and periodically solves the local stage game, using a pure Nash equilibrium when one exists and fictitious play otherwise.",
          images: images.carZeroSum,
          details: [
            "Enumerates the 5-by-5 grid state space as joint positions (x1, y1, x2, y2).",
            "Uses a Q-table where each state stores a 4-by-4 joint-action payoff matrix.",
            "Rewards include a grid-position reward, living cost, stay penalty, and collision penalty.",
            "Solves local zero-sum stage games with pure equilibria when available and fictitious play otherwise.",
            "Refreshes cached policies periodically during Q-iteration instead of resolving every state on every pass.",
          ],
        },
        {
          name: "General-Sum Solver",
          description: "A more realistic twist: both players now have their own payoff matrices. Player 1 is rewarded for collision or reduced distance, while Player 2 is rewarded for avoiding collision, increasing distance, and its own grid-position reward.",
          images: images.carGenSum,
          details: [
            "Uses a larger 10-by-10 grid and stores separate Q1 and Q2 tables for each state.",
            "Player 1 is rewarded for closing distance or colliding, while Player 2 is rewarded for keeping distance.",
            "Searches for pure Nash equilibria in each local stage game.",
            "Falls back to independent best responses when no pure equilibrium is found.",
          ],
        },
        {
          name: "DQN Planning",
          description: "Instead of storing a Q-table for every state, this version trains a neural network to approximate Q-values over the 16 joint actions. It uses the environment model to compute Bellman targets by simulating transitions, then extracts a minimax policy from the learned Q matrix.",
          images: images.carDQNPlanning,
          details: [
            "Normalizes the four position coordinates as the neural-network input.",
            "Uses a 4 -> 64 -> 64 -> 16 network, where the 16 outputs reshape into a 4-by-4 joint-action matrix.",
            "Chooses Player 1's action by maximizing the minimum value over Player 2's response.",
            "Trains from replay with a target network, epsilon-greedy exploration, and a minimax Bellman backup.",
          ],
        }
      ],
    },
    {
      title: "Dog Game",
      intro: "Two players move around a continuous 2D space. Their midpoint is a virtual 'dog.' Each player is trying to steer the dog toward their own house. This is a cooperative-competitive setting, they share the dog but have different goals.",
      info: [
        {
          label: "Environment",
          text: "A continuous two-player control problem where the dog is defined by the midpoint between the two agents.",
        },
        {
          label: "State",
          text: "The state stores both agents' positions and the two fixed house positions; the dog's midpoint is computed from the agents after each move.",
        },
        {
          label: "Actions",
          text: "Agents choose movement directions. The 8-direction and 16-direction versions test how action resolution affects learned behavior.",
        },
        {
          label: "Goal",
          text: "Each player wants the shared dog to move toward their own house, creating a mixed cooperative and competitive learning problem.",
        },
      ],
      subs: [
        {
          name: "Nash Solver",
          description: "Each agent has a joint-action Q-network that outputs a K-by-K payoff matrix over both players' actions. During action selection and target computation, the code solves an approximate general-sum Nash equilibrium using repeated best responses.",
          images: images.dogNash,
          details: [
            "Uses separate networks for Player 1 and Player 2, each mapping the 8-value state to a K-by-K joint-action matrix.",
            "Computes rewards as the negative distance from the dog midpoint to each player's house.",
            "Represents movement as full-step and half-step directions, with an optional stay action.",
            "Uses iterated best responses to approximate the general-sum Nash backup during training.",
            "Uses an LRU cache for rounded next states to avoid repeatedly solving the same Nash backup.",
          ],
          analysis: "However, because a solver is needed at every step, training is very slow. The agents learn a decent strategy but it's not as polished as the DQN versions below. My efforts would then focus on the learning approach, which is more scalable and ultimately more interesting since it doesn't assume access to the environment model or a solver.",
          
        },
        {
          name: "Learning (8 directions)",
          description: "Each player gets their own independent Q-network and learns on their own, no joint action matrix, no game theory solver. Despite the simplicity, they still learn to cooperate because their rewards are coupled through the dog's position. Soft target network updates keep training stable.",
          images: images.dogLearning8,
          details: [
            "Uses independent Q-networks for the two agents, each outputting values only for its own action choices.",
            "Removes the equilibrium solver from the training loop.",
            "Uses 8 base directions, plus half-step versions and a stay action in the 8-direction experiment.",
            "Updates each agent from replay using Smooth L1 loss, clipped gradients, and soft target-network updates.",
          ],
          analysis: "With 8 directions the agents learn a coarse policy. Movement is visibly blocky and the dog tends to overshoot the target house before correcting.",
        },
        {
          name: "Learning (16 directions)",
          description: "Each player gets their own independent Q-network and learns on their own, no joint action matrix, no game theory solver. Despite the simplicity, they still learn to cooperate because their rewards are coupled through the dog's position. Soft target network updates keep training stable.",
          images: images.dogLearning16,
          details: [
            "Uses 16 base directions, plus half-step versions and a stay action in the 16-direction experiment.",
            "Keeps the same independent DQN structure as the 8-direction version.",
            "Trains longer with slower epsilon decay so the finer action space has time to stabilize.",
            "Improves trajectory quality by giving each player finer control.",
          ],
          analysis: "16 directions gives much smoother trajectories. The agents develop a more refined cooperative strategy, with the dog taking more direct paths to the target.",

        },
      ],
    },
    {
      title: "Territory War (DQN)",
      intro: "Two DQN agents take turns moving through a grid and claiming territory. Each move either expands into an empty cell or, when boxed in, routes through owned cells toward the nearest reachable frontier. The winner is the player with more claimed cells when the board is exhausted or the move limit is reached.",
      info: [
        {
          label: "Environment",
          text: "A grid-based area-control game with red and blue agents, optional walls, and a board that records empty, red, blue, and wall cells.",
        },
        {
          label: "State",
          text: "The base state includes both player positions, four local neighbor-cell values, distance to the opponent, and distance to the nearest frontier.",
        },
        {
          label: "Actions",
          text: "Each agent chooses one of four moves: up, right, down, or left. Illegal moves into walls, board edges, or enemy cells are masked out.",
        },
        {
          label: "Reward",
          text: "Agents receive reward for claiming empty cells, small penalties for moving through owned cells, frontier-progress shaping, territory advantage shaping, and terminal win/loss/draw bonuses.",
        },
        {
          label: "Limitation",
          text: "The DQN setup mostly rewards expansion, so agents do not explicitly learn adversarial tactics like blocking, cutting off paths, or sacrificing short-term cells to limit the other player.",
        },
        {
          label: "Future Work",
          text: "A stronger version could use CNNs over the board state and add objectives that reward blocking, cutting off regions, and predicting how the opponent will expand.",
        },
      ],
      subs: [
        {
          name: "Independent DQN, No Walls",
          description: "This version trains separate DQN agents for red and blue on a smaller training grid, then rolls the learned greedy policies out on the larger board. Legal actions are constrained so agents expand whenever an adjacent empty cell is available, and otherwise move along a shortest route back to reachable empty territory. That makes the behavior good for claiming space, but weak at intentionally interfering with the opponent.",
          images: images.territoryStandardNoWalls,
          details: [
            "Uses one online and one target network per player.",
            "Masks illegal next actions during DQN target computation.",
            "Encodes local neighbor cells, opponent distance, and distance to the nearest frontier as state features.",
            "Routes boxed-in agents through owned cells toward the closest reachable empty frontier.",
            "Randomizes side assignment and first player so policies do not memorize one opening.",
          ],
          analysis: "Without walls, the learned policy expands across an open board, so the main pressure is how efficiently each agent reaches and claims available frontier cells. Because there is no direct blocking objective, the agents usually race for space instead of planning denial moves.",
        },
        {
          name: "Independent DQN, With Walls",
          description: "This uses the same independent DQN setup, but enables the wall layout from the environment. The wall blocks movement and claiming, changing which frontiers are reachable and forcing the agents to work around separated regions.",
          images: images.territoryStandardWalls,
          details: [
            "Keeps the same state features, rewards, and DQN architecture as the no-wall version.",
            "Masks wall cells out of legal movement.",
            "Places a vertical wall segment in the board so reachable frontiers and routes change during rollout.",
            "Tests whether the local frontier policy still works when the board is split by obstacles.",
          ],
          analysis: "The wall creates more structured expansion fronts. Agents can no longer treat the map as one open territory, so routing back to reachable empty cells matters more. Still, the competition is mostly indirect: walls constrain movement, but the agents are not learning to create those constraints against each other.",
        },
        {
          name: "Chunk Feature, No Walls",
          description: "This variant adds global information about the largest connected empty region. The state includes the center of that region and the active player's distance to it, giving the agent a signal for where the most valuable remaining territory is concentrated.",
          images: images.territoryChunkNoWalls,
          details: [
            "Finds connected empty components with a breadth-first search.",
            "Adds the largest chunk center and chunk-distance feature to the DQN input.",
            "Highlights the largest unclaimed region during rollout so the effect of the global feature is visible.",
            "Uses the same reward structure and legal-action rules as the base territory game.",
          ],
          analysis: "On an open board, the chunk feature is meant to reduce shortsighted expansion by pointing the agent toward large remaining regions instead of only the nearest frontier.",
        },
        {
          name: "Chunk Feature, With Walls",
          description: "This combines the wall environment with the biggest-unclaimed-chunk state feature. Because walls split the board into connected regions, the chunk signal can help identify which reachable empty region is most strategically important.",
          images: images.territoryChunkWalls,
          details: [
            "Computes the largest connected empty component while respecting occupied and wall cells.",
            "Adds chunk center and chunk-distance inputs on top of the base state.",
            "Combines local legal-action masking with a coarse global target for the remaining empty space.",
            "Keeps the same independent DQN training loop and terminal win/loss/draw shaping.",
          ],
          analysis: "With walls enabled, the chunk feature has a clearer role: it gives the policy a coarse global target when obstacles divide the remaining territory.",
        },
      ],
      takeaway: "The next step would be to make the agents reason about the board more spatially and adversarially. A CNN-based policy could see territory shapes, walls, bottlenecks, and frontier patterns directly instead of relying only on hand-built local features. The reward could also include competitive signals for denying access to large regions, creating chokepoints, or reducing the opponent's reachable empty cells.",
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
            How do you teach two AI agents to play a game against each other, or with each other, when neither knows what the other will do? This project explores that question through two custom environments and four different approaches, from classical game theory to deep reinforcement learning.
          </p>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: width_allowed, margin: "0 auto", padding: "60px 24px 120px" }}>
        <section style={{ marginBottom: 80 }}>
          <div style={{ borderLeft: `4px solid ${blue}`, paddingLeft: 16, marginBottom: 20 }}>
            <h2 style={{ fontSize: 24, fontWeight: 700, margin: "0 0 8px", color: "#111" }}>
              Overview
            </h2>
            <p style={{ fontSize: 15, color: "#555", lineHeight: 1.7, margin: 0 }}>
              The implementations move from exact game-solving methods toward neural approximations and larger learned policies.
            </p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
            {overview.map((item) => (
              <p key={item} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 14, fontSize: 14, color: "#555", lineHeight: 1.65, margin: 0 }}>
                {item}
              </p>
            ))}
          </div>
        </section>
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
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginBottom: 28 }}>
              {project.info.map((item) => (
                <div key={item.label} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 14 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: blue, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 6 }}>
                    {item.label}
                  </div>
                  <p style={{ fontSize: 14, color: "#555", lineHeight: 1.6, margin: 0 }}>
                    {item.text}
                  </p>
                </div>
              ))}
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
                {sub.details && (
                  <ul style={{ color: "#555", fontSize: 14, lineHeight: 1.65, margin: "0 0 20px", paddingLeft: 22 }}>
                    {sub.details.map((detail) => (
                      <li key={detail}>{detail}</li>
                    ))}
                  </ul>
                )}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "start" }}>
                  {sub.images.map(([src, cropBottom, cropTop, scale], i) => (
                    <ProjectImage
                      key={i}
                      src={src}
                      alt={`${sub.name} ${i + 1}`}
                      cropBottom={cropBottom}
                      cropTop={cropTop}
                      scale={scale}
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
            {project.takeaway && (
              <p style={{ fontSize: 15, color: "#555", lineHeight: 1.75, margin: "-20px 0 56px", borderTop: "1px solid #e5e7eb", paddingTop: 16 }}>
                {project.takeaway}
              </p>
            )}

            {pi < projects.length - 1 && (
              <hr style={{ border: "none", borderTop: "1px solid #e5e7eb", margin: "0 0 80px" }} />
            )}
          </div>
        ))}
        <section>
          <div style={{ borderLeft: `4px solid ${blue}`, paddingLeft: 16, marginBottom: 20 }}>
            <h2 style={{ fontSize: 24, fontWeight: 700, margin: "0 0 8px", color: "#111" }}>
              Limitations & Future Work
            </h2>
            <p style={{ fontSize: 15, color: "#555", lineHeight: 1.7, margin: 0 }}>
              The experiments work, but they also show where the current approaches start to strain.
            </p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
            <div>
              <h3 style={{ fontSize: 18, fontWeight: 600, margin: "0 0 12px", color: "#111" }}>
                Limitations
              </h3>
              <ul style={{ color: "#555", fontSize: 14, lineHeight: 1.65, margin: 0, paddingLeft: 22 }}>
                {limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 style={{ fontSize: 18, fontWeight: 600, margin: "0 0 12px", color: "#111" }}>
                Future Work
              </h3>
              <ul style={{ color: "#555", fontSize: 14, lineHeight: 1.65, margin: 0, paddingLeft: 22 }}>
                {futureWork.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

