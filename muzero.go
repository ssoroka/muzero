package main

import (
	"math"
)

// originally this was built with numpy and tensorflow.
// I've made some efforts to remove numpy where I could and replace the functions with Go versions, see numpy.go for those.
//
// Missing function implementations collected from TODO below:
// GetGlobalStep
// MomentumOptimizer
// StopGradient
// softmaxCrossEntropyWithLogits
// l2loss
// optimizer (object)

const MaximumFloatValue = math.MaxFloat64

type KnownBounds struct {
	min float64
	max float64
}

// MinMaxStats holds the min-max values of the tree.
type MinMaxStats struct {
	Minimum float64
	Maximum float64
}

func NewMinMaxStats(min, max float64) MinMaxStats {
	s := MinMaxStats{
		Minimum: -MaximumFloatValue,
		Maximum: MaximumFloatValue,
	}
	if min != 0 {
		s.Minimum = min
	}
	if max != 0 {
		s.Maximum = max
	}
	return s
}

func (s *MinMaxStats) Update(val float64) {
	s.Maximum = math.Max(s.Maximum, val)
	s.Minimum = math.Min(s.Minimum, val)
}

func (s *MinMaxStats) Normalize(val float64) float64 {
	if s.Maximum > s.Minimum {
		// We normalize only when we have set the maximum and minimum values.
		return (val - s.Minimum) / (s.Maximum - s.Minimum)
	}
	return val
}

type MuZeroConfig struct {
	ActionSpaceSize           int
	NumActors                 int
	VisitSoftmaxTemperatureFn VisitSoftmaxTemperatureFunc
	MaxMoves                  int
	NumSimulations            int
	Discount                  float64
	RootDirichletAlpha        float64
	RootExplorationFraction   float64
	PbCBase                   int
	PbCInit                   float64
	KnownBounds               KnownBounds
	TrainingSteps             int
	CheckpointInterval        int
	WindowSize                int
	BatchSize                 int
	NumUnrollSteps            int
	TdSteps                   int
	WeightDecay               float64
	Momentum                  float64
	LrInit                    float64
	LrDecayRate               float64
	LrDecaySteps              float64
}

func NewMuZeroConfig(
	actionSpaceSize int,
	maxMoves int,
	discount float64,
	dirichletAlpha float64,
	numSimulations int,
	batchSize int,
	tdSteps int,
	numActors int,
	lrInit float64,
	lrDecaySteps float64,
	visitSoftmaxTemperatureFn VisitSoftmaxTemperatureFunc,
	knownBounds *KnownBounds,
) *MuZeroConfig {
	cfg := &MuZeroConfig{}
	// ### Self-Play
	cfg.ActionSpaceSize = actionSpaceSize
	cfg.NumActors = numActors

	cfg.VisitSoftmaxTemperatureFn = visitSoftmaxTemperatureFn
	cfg.MaxMoves = maxMoves
	cfg.NumSimulations = numSimulations
	cfg.Discount = discount

	// Root prior exploration noise.
	cfg.RootDirichletAlpha = dirichletAlpha
	cfg.RootExplorationFraction = 0.25

	// UCB formula
	cfg.PbCBase = 19652
	cfg.PbCInit = 1.25

	// If we already have some information about which values occur in the
	// environment, we can use them to initialize the rescaling.
	// This is not strictly necessary, but establishes identical behaviour to
	// AlphaZero in board games.
	if knownBounds != nil {
		cfg.KnownBounds = *knownBounds
	}

	// ### Training
	cfg.TrainingSteps = int(1000e3)
	cfg.CheckpointInterval = int(1e3)
	cfg.WindowSize = int(1e6)
	cfg.BatchSize = batchSize
	cfg.NumUnrollSteps = 5
	cfg.TdSteps = tdSteps

	cfg.WeightDecay = 1e-4
	cfg.Momentum = 0.9

	// Exponential learning rate schedule
	cfg.LrInit = lrInit
	cfg.LrDecayRate = 0.1
	cfg.LrDecaySteps = lrDecaySteps
	return cfg
}

func (cfg *MuZeroConfig) NewGame() *Game {
	return NewGame(cfg.ActionSpaceSize, cfg.Discount)
}

type VisitSoftmaxTemperatureFunc func(numMoves, trainingSteps int) float64

func visitSoftmaxTemperature(numMoves, trainingSteps int) float64 {
	if numMoves < 30 {
		return 1.0
	}
	return 0.0
}

func makeBoardGameConfig(
	actionSpaceSize int,
	maxMoves int,
	dirichletAlpha float64,
	lrInit float64,
) *MuZeroConfig {
	return NewMuZeroConfig(
		actionSpaceSize,
		maxMoves,
		1.0,
		dirichletAlpha,
		800,
		2048,
		maxMoves, // Always use Monte Carlo return.
		3000,
		lrInit,
		400e3,
		visitSoftmaxTemperature,
		&KnownBounds{-1, 1})
}

func makeGoConfig() *MuZeroConfig {
	return makeBoardGameConfig(362, 722, 0.03, 0.01)
}

func makeChessConfig() *MuZeroConfig {
	return makeBoardGameConfig(4672, 512, 0.3, 0.1)
}

func makeShogiConfig() *MuZeroConfig {
	return makeBoardGameConfig(11259, 512, 0.15, 0.1)
}

func atariVisitSoftmaxTemperature(num_moves, training_steps int) float64 {
	if training_steps < 500e3 {
		return 1.0
	}
	if training_steps < 750e3 {
		return 0.5
	}
	return 0.25
}

func makeAtariConfig() *MuZeroConfig {
	return NewMuZeroConfig(
		18,                           // actionSpaceSize
		27000,                        // maxMoves // Half an hour at action repeat 4.
		0.997,                        // discount
		0.25,                         // dirichletAlpha
		50,                           // numSimulations
		1024,                         // batchSize
		10,                           // tdSteps
		350,                          // numActors
		0.05,                         // lrInit
		350e3,                        // lrDecaySteps
		atariVisitSoftmaxTemperature, // visitSoftmaxTemperatureFn
		nil,                          // knownBounds
	)
}

type Action int

type Player struct{}

type Node struct {
	VisitCount  int
	ToPlay      *Player
	Prior       float64
	ValueSum    float64
	Children    map[Action]Node
	HiddenState []float64
	Reward      float64
}

func NewNode(prior float64) Node {
	return Node{
		VisitCount: 0,
		ToPlay:     nil,
		Prior:      prior,
		ValueSum:   0,
		Children:   map[Action]Node{},
		Reward:     0,
	}
}

func (n *Node) Expanded() bool {
	return len(n.Children) > 0
}

func (n *Node) Value() float64 {
	if n.VisitCount == 0 {
		return 0
	}
	return n.ValueSum / float64(n.VisitCount)
}

///Simple history container used inside the search.
///
///  Only used to keep track of the actions executed.
///
type ActionHistory struct {
	History         []Action
	ActionSpaceSize int
}

func NewActionHistory(history []Action, actionSpaceSize int) *ActionHistory {
	return &ActionHistory{
		History:         history,
		ActionSpaceSize: actionSpaceSize,
	}
}

func (ah *ActionHistory) Clone() *ActionHistory {
	h := make([]Action, len(ah.History))
	for i := range h {
		h[i] = ah.History[i]
	}
	return NewActionHistory(h, ah.ActionSpaceSize)
}

func (ah *ActionHistory) AddAction(action Action) {
	ah.History = append(ah.History, action)
}

func (ah *ActionHistory) LastAction() Action {
	return ah.History[len(ah.History)-1]
}

func (ah *ActionHistory) ActionSpace() []Action {
	return makeActionSpace(ah.ActionSpaceSize)
}

func makeActionSpace(size int) []Action {
	a := make([]Action, size)
	for i := 0; i < len(a); i++ {
		a[i] = Action(i)
	}
	return a
}

func (ah *ActionHistory) ToPlay() *Player {
	return &Player{}
}

///The environment MuZero is interacting with.///
type Environment struct{}

func (e Environment) Step(action Action) (reward float64) {
	// take some action
	return 0
}

///A single episode of interaction with the environment.///
type Game struct {
	Environment     Environment
	History         []Action
	Rewards         []float64
	ChildVisits     []float64
	RootValues      []float64
	ActionSpaceSize int
	Discount        float64
}

func NewGame(actionSpaceSize int, discount float64) *Game {
	return &Game{
		Environment:     Environment{}, // Game specific environment.
		History:         []Action{},
		Rewards:         []float64{},
		ChildVisits:     []float64{},
		RootValues:      []float64{},
		ActionSpaceSize: actionSpaceSize,
		Discount:        discount,
	}
}

// Game specific termination rules. true when game is over
func (g *Game) Terminal() bool {
	return true
}

// Game specific calculation of legal actions.
func (g *Game) LegalActions() []Action {
	return []Action{}
}

func (g *Game) Apply(action Action) {
	reward := g.Environment.Step(action)
	g.Rewards = append(g.Rewards, reward)
	g.History = append(g.History, action)
}

func (g *Game) StoreSearchStatistics(root *Node) {
	sumVisits := 0
	for _, child := range root.Children {
		sumVisits += child.VisitCount
	}
	actionSpace := makeActionSpace(g.ActionSpaceSize)
	for _, a := range actionSpace {
		if child, ok := root.Children[a]; ok {
			g.ChildVisits = append(g.ChildVisits, float64(child.VisitCount)/float64(sumVisits))
		}
	}
	g.RootValues = append(g.RootValues, root.Value())
}

type Image interface{}

// Game specific feature planes.
func (g *Game) MakeImage(stateIndex int) Image {
	return nil // some kind of vector.
}

func (g *Game) MakeTarget(stateIndex, numUnrollSteps, tdSteps int, toPlay *Player) []Target {
	targets := []Target{}
	// The value target is the discounted root value of the search tree N steps
	// into the future, plus the discounted sum of all rewards until then.
	for currentIndex := stateIndex; currentIndex < stateIndex+numUnrollSteps+1; currentIndex++ {
		bootstrapIndex := currentIndex + tdSteps
		value := float64(0)
		if bootstrapIndex < len(g.RootValues) {
			value = g.RootValues[bootstrapIndex] * math.Pow(g.Discount, float64(tdSteps))
		}

		for i, reward := range g.Rewards[currentIndex:bootstrapIndex] {
			value += reward * math.Pow(g.Discount, float64(i))
		}

		// For simplicity the network always predicts the most recently received
		// reward, even for the initial representation network where we already
		// know this reward.
		lastReward := float64(0)
		if currentIndex > 0 && currentIndex <= len(g.Rewards) {
			lastReward = g.Rewards[currentIndex-1]
		}

		if currentIndex < len(g.RootValues) {
			targets = append(targets, Target{value, lastReward, []float64{g.ChildVisits[currentIndex]}})
		} else {
			// States past the end of games are treated as absorbing states.
			targets = append(targets, Target{0, lastReward, []float64{}})
		}
	}
	return targets
}

// TODO: I think this is which player's turn it is.
func (g *Game) ToPlay() *Player {
	return &Player{}
}

func (g *Game) ActionHistory() *ActionHistory {
	return NewActionHistory(g.History, g.ActionSpaceSize)
}

type Target struct {
	Value       float64
	LastReward  float64
	ChildVisits []float64 // TODO: is this policy?
}

type ReplayBuffer struct {
	WindowSize int
	BatchSize  int
	Buffer     []Game
}

func NewReplayBuffer(cfg MuZeroConfig) *ReplayBuffer {
	return &ReplayBuffer{
		WindowSize: cfg.WindowSize,
		BatchSize:  cfg.BatchSize,
		Buffer:     []Game{},
	}
}

func (b *ReplayBuffer) SaveGame(game *Game) {
	if len(b.Buffer) > b.WindowSize {
		b.Buffer = b.Buffer[1:]
	}
	// todo: clone
	b.Buffer = append(b.Buffer, *game)
}

func (b *ReplayBuffer) SampleBatch(numUnrollSteps, tdSteps int) (result []BatchItem) {
	result = make([]BatchItem, b.BatchSize)
	for i := 0; i < b.BatchSize; i++ {
		game := b.SampleGame()
		gamePos := b.SamplePosition(game)

		item := BatchItem{
			Image:   game.MakeImage(gamePos),
			Actions: game.History[i : i+numUnrollSteps],
			Targets: game.MakeTarget(gamePos, numUnrollSteps, tdSteps, game.ToPlay()),
		}
		result[i] = item
	}
	return result
}

// TODO: Sample game from buffer either uniformly or according to some priority.
func (b *ReplayBuffer) SampleGame() Game {
	return b.Buffer[0]
}

// TODO: Sample position from game either uniformly or according to some priority.
func (b *ReplayBuffer) SamplePosition(game Game) int {
	return -1
}

type BatchItem struct {
	Image   Image
	Actions []Action
	Targets []Target
}

// type PolicyLogitsType map[Action]float64

type NetworkOutput struct {
	Value        float64
	Reward       float64
	PolicyLogits map[Action]float64
	HiddenState  []float64
}

type Network struct {
	TrainingSteps int
	Weights       []Tensor
	// TODO: seems like things might be missing here.
}

// representation + prediction function
func (n *Network) InitialInference(image Image) *NetworkOutput {
	return &NetworkOutput{0, 0, map[Action]float64{}, []float64{}}
}

// dynamics + prediction function
func (n *Network) RecurrentInference(hiddenState []float64, action Action) *NetworkOutput {
	return &NetworkOutput{0, 0, map[Action]float64{}, []float64{}}
}

// Returns the weights of this network.
func (n *Network) GetWeights() []Tensor {
	return []Tensor{}
}

// // How many steps / batches the network has been trained for.
// func (n *Network) GetTrainingSteps() int {
// 	return 0
// }

type Tensor []float64 // ? interface{} // a placeholder for whatever this is going to end up being. See weights.

type SharedStorage struct {
	Networks map[int]Network
	LastStep int
}

func (s *SharedStorage) LatestNetwork() *Network {
	if s.Networks == nil {
		// policy -> uniform, value -> 0, reward -> 0
		return makeUniformNetwork()
	}
	network := s.Networks[s.LastStep]
	return &network
}

func (s *SharedStorage) SaveNetwork(step int, network *Network) {
	s.Networks[step] = *network // TODO: clone tensor
	s.LastStep = step
}

// ##### End Helpers ########
// ##########################

// MuZero training is split into two independent parts: Network training and
// self-play data generation.
// These two parts only communicate by transferring the latest network checkpoint
// from the training to the self-play, and the finished games from the self-play
// to the training.
func muzero(config MuZeroConfig) *Network {
	storage := &SharedStorage{}
	replayBuffer := NewReplayBuffer(config)

	for i := 0; i < config.NumActors; i++ {
		launchJob(runSelfplay, config, storage, replayBuffer)
	}

	trainNetwork(config, storage, replayBuffer)

	return storage.LatestNetwork()
}

// ##################################
// ####### Part 1: Self-Play ########

// runSelfPlay and trainNetwork are SelfPlayJobs
type SelfPlayJob func(config MuZeroConfig, storage *SharedStorage, replayBuffer *ReplayBuffer)

// Each self-play job is independent of all others; it takes the latest network
// snapshot, produces a game and makes it available to the training job by
// writing it to a shared replay buffer.
func runSelfplay(config MuZeroConfig, storage *SharedStorage, replayBuffer *ReplayBuffer) {
	for true {
		network := storage.LatestNetwork()
		game := playGame(config, network)
		replayBuffer.SaveGame(game)
	}
}

// Each game is produced by starting at the initial board position, then
// repeatedly executing a Monte Carlo Tree Search to generate moves until the end
// of the game is reached.
func playGame(config MuZeroConfig, network *Network) *Game {
	game := config.NewGame()

	for !game.Terminal() && len(game.History) < config.MaxMoves {
		// At the root of the search tree we use the representation function to
		// obtain a hidden state given the current observation.
		root := &Node{}
		currentObservation := game.MakeImage(-1)
		expandNode(root, game.ToPlay(), game.LegalActions(), network.InitialInference(currentObservation))
		addExplorationNoise(config, root)

		// We then run a Monte Carlo Tree Search using only action sequences and the
		// model learned by the network.
		runMcts(config, root, game.ActionHistory(), network)
		action := selectAction(config, len(game.History), root, network)
		game.Apply(action)
		game.StoreSearchStatistics(root)
	}
	return game
}

// Core Monte Carlo Tree Search algorithm.
// To decide on an action, we run N simulations, always starting at the root of
// the search tree and traversing the tree according to the UCB formula until we
// reach a leaf node.
func runMcts(config MuZeroConfig, root *Node, actionHistory *ActionHistory, network *Network) {
	minMaxStats := NewMinMaxStats(config.KnownBounds.min, config.KnownBounds.max) // todo: could make these the same

	for i := 0; i < config.NumSimulations; i++ {
		history := actionHistory.Clone()
		node := root
		searchPath := []*Node{node}

		for node.Expanded() {
			action, node := selectChildWithHighestScore(config, node, minMaxStats)
			history.AddAction(action)
			searchPath = append(searchPath, node)
		}

		// Inside the search tree we use the dynamics function to obtain the next
		// hidden state given an action and the previous hidden state.
		parent := searchPath[len(searchPath)-2]
		networkOutput := network.RecurrentInference(parent.HiddenState, history.LastAction())
		expandNode(node, history.ToPlay(), history.ActionSpace(), networkOutput)

		backpropagate(searchPath, networkOutput.Value, history.ToPlay(),
			config.Discount, minMaxStats)
	}
}

func selectAction(config MuZeroConfig, numMoves int, node *Node, network *Network) Action {
	temperature := config.VisitSoftmaxTemperatureFn(numMoves, network.TrainingSteps)
	_, action := softmaxSample(node.Children, temperature)
	return action
}

// Select the child with the highest UCB score.
func selectChildWithHighestScore(config MuZeroConfig, node *Node, minMaxStats MinMaxStats) (selectedAction Action, selectedNode *Node) {
	score := float64(-1)
	for action, child := range node.Children {
		s := ucbScore(config, node, &child, minMaxStats)
		if s > score {
			score = s
			selectedAction = action
			*selectedNode = child
		}
	}
	return selectedAction, selectedNode
}

// The score for a node is based on its value, plus an exploration bonus based on
// the prior.
func ucbScore(config MuZeroConfig, parent, child *Node, minMaxStats MinMaxStats) float64 {
	pbC := math.Log(float64(parent.VisitCount+config.PbCBase+1)/float64(config.PbCBase)) + config.PbCInit
	pbC *= math.Sqrt(float64(parent.VisitCount)) / float64(child.VisitCount+1)

	priorScore := pbC * child.Prior
	valueScore := float64(0)
	if child.VisitCount > 0 {
		valueScore = child.Reward + config.Discount*minMaxStats.Normalize(child.Value())
	}
	return priorScore + valueScore
}

// We expand a node using the value, reward and policy prediction obtained from
// the neural network.
func expandNode(node *Node, toPlay *Player, actions []Action, networkOutput *NetworkOutput) {
	node.ToPlay = toPlay
	node.HiddenState = networkOutput.HiddenState
	node.Reward = networkOutput.Reward
	policy := map[Action]float64{}
	policySum := float64(0)
	for _, a := range actions {
		v := math.Exp(networkOutput.PolicyLogits[a])
		policy[a] = v
		policySum += v
	}
	for action, p := range policy {
		node.Children[action] = NewNode(p / policySum)
	}
}

// At the end of a simulation, we propagate the evaluation all the way up the
// tree to the root.
func backpropagate(searchPath []*Node, value float64, toPlay *Player, discount float64, minMaxStats MinMaxStats) {
	var node *Node
	for i := len(searchPath) - 1; i >= 0; i-- {
		node = searchPath[i]
		if node.ToPlay == toPlay {
			node.ValueSum += value
		} else {
			node.ValueSum -= value
		}
		node.VisitCount++
		minMaxStats.Update(node.Value())

		value = node.Reward + discount*value
	}
}

// At the start of each search, we add dirichlet noise to the prior of the root
// to encourage the search to explore new actions.
func addExplorationNoise(config MuZeroConfig, node *Node) {
	actions := make([]Action, len(node.Children))
	i := 0
	for act := range node.Children {
		actions[i] = act
		i++
	}
	noise := make([]float64, len(actions))
	for i := range noise {
		noise[i] = config.RootDirichletAlpha
	}
	noise = dirichlet(noise)
	frac := config.RootExplorationFraction
	for i, a := range actions {
		n := node.Children[a]
		n.Prior = n.Prior*(1.0-frac) + noise[i]*frac
		node.Children[a] = n
	}
}

// ######### End Self-Play ##########
// ##################################

// ##################################
// ####### Part 2: Training #########

func trainNetwork(config MuZeroConfig, storage *SharedStorage, replayBuffer *ReplayBuffer) {
	network := &Network{}
	learningRate := config.LrInit * math.Pow(config.LrDecayRate, (train.GetGlobalStep()/config.LrDecaySteps)) // TODO: needs attention.
	optimizer := train.MomentumOptimizer(learningRate, config.Momentum)                                       // TODO: needs attention.

	for i := 0; i < config.TrainingSteps; i++ {
		if i%config.CheckpointInterval == 0 {
			storage.SaveNetwork(i, network)
		}
		batch := replayBuffer.SampleBatch(config.NumUnrollSteps, config.TdSteps)
		updateWeights(optimizer, network, batch, config.WeightDecay)
	}
	storage.SaveNetwork(config.TrainingSteps, network)
}

///Scales the gradient for the backward pass.///
func scaleGradient(tensor Tensor, scale float64) Tensor {
	return tensor*scale + tf.StopGradient(tensor)*(1.0-scale) // TODO: needs attention.
}

type Prediction struct {
	GradientScale float64
	Value         float64
	Reward        float64
	PolicyLogits  map[Action]float64
}

// TODO: needs attention.
func updateWeights(optimizer train.Optimizer, network *Network, batch []BatchItem, weightDecay float64) {
	loss := float64(0)
	for _, item := range batch {
		// Initial step, from the real observation.
		// value, reward, policyLogits, hiddenState = network.InitialInference(image)
		output := network.InitialInference(item.Image)
		predictions := []Prediction{{1.0, output.Value, output.Reward, output.PolicyLogits}}

		hiddenState := output.HiddenState
		// Recurrent steps, from action and previous hidden state.
		for _, action := range item.Actions {
			// value, reward, policyLogits, hiddenState = network.recurrentInference( hiddenState, action)
			out := network.RecurrentInference(hiddenState, action)
			// predictions.append((1.0 / len(actions), value, reward, policyLogits))
			predictions = append(predictions, Prediction{1.0 / float64(len(item.Actions)), out.Value, out.Reward, out.PolicyLogits})

			hiddenState = scaleGradient(out.HiddenState, 0.5)
		}
		// for prediction, target in zip(predictions, targets)
		for i, prediction := range predictions {
			target := item.Targets[i]
			// gradientScale, value, reward, policyLogits = prediction
			// targetValue, targetReward, targetPolicy = target

			predictedLoss := (scalarLoss(prediction.Value, target.Value) +
				scalarLoss(prediction.Reward, target.LastReward) +
				// logits = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]
				// labels = [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]
				// tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
				// tf.nn.softmaxCrossEntropyWithLogits(logits=policyLogits, labels=targetPolicy))
				tf.nn.softmaxCrossEntropyWithLogits(target.ChildVisits, prediction.PolicyLogits)) // TODO: needs attention.

			loss += scaleGradient(predictedLoss, prediction.GradientScale)
		}

	}
	for _, weights := range network.GetWeights() {
		loss += weightDecay * tf.nn.L2Loss(weights) // TODO: needs attention.
	}

	optimizer.Minimize(loss)
}

// MSE in board games, cross entropy between categorical values in Atari.
func scalarLoss(prediction, target float64) float64 {
	return -1
}

// ######### End Training ###########
// ##################################

// ################################################################################
// ############################# End of pseudocode ################################
// ################################################################################

// Stubs to make the typechecker happy.
func softmaxSample(distribution map[Action]Node, temperature float64) (maybeThisIsNode *Node, action Action) {
	for action, node := range distribution {
		return &node, action
	}
	return nil, 0
}

func launchJob(job SelfPlayJob, config MuZeroConfig, storage *SharedStorage, replayBuffer *ReplayBuffer) {
	job(config, storage, replayBuffer)
}

func makeUniformNetwork() *Network {
	return &Network{}
}
