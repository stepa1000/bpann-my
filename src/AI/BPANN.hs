-----------------------------------------------------------------------------
-- |
-- Module      :  BPANN
-- Copyright   :  (c) Robert Steuck 2011
-- License     :  AllRightsReserved
--
-- Maintainer  :  robert.steuck@gmail.com
-- Stability   :  experimental
-- Portability :  portable
--
-- Basic backpropagation neuronal network
-- inspired by hnn

module AI.BPANN where

import Data.List
import Data.List.Split
import Data.Maybe
import System.Random

-- ** Types for computation
type ALayer a = [(Neuron,a)] -- Das erste Neuron ist immer das BIAS Neuron

type ANetwork a = [ALayer a]

type Network = ANetwork ()

-- |information generated during a simple forward pass
-----------------------------------------------------------
data ForwardPassInfo = FPInfo {
-- |output
-----------------------------------------------------------
  o :: Double,
-- |sum of weighted inputs
-----------------------------------------------------------
  net :: Double, -- Summe der Gewichteten Eingaben
-- |inputs
-----------------------------------------------------------
  xs :: [Double] -- Ungewichtete Eingaben
} deriving Show

-- |the neuron
-----------------------------------------------------------
data Neuron = Neuron {
-- |input weights
-----------------------------------------------------------
  ws :: [Double],
-- |activation function
-----------------------------------------------------------
  fun :: (Double -> Double),
-- |first derivation of the activation function
-----------------------------------------------------------
  fun' :: (Double -> Double) -- 1. Ableitung der Aktivierungsfunktion
}

instance Show Neuron where
  show (Neuron ws _ _) = 
    "Neuron: ws=" ++ (show ws)

-- ** Types for serialisation
type PackedNeuron = [Double]

-- ** Activation functions
-- |1/(1+e^(-x))
-----------------------------------------------------------
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1 + exp (-x)) 

-- |first derivation
-----------------------------------------------------------
sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- ** Network creation
type NeuronCreator = PackedNeuron -> Neuron

sigmoidNeuron :: PackedNeuron -> Neuron
sigmoidNeuron ws = Neuron ws sigmoid sigmoid'

-- |activation function is 'id'
-----------------------------------------------------------
outputNeuron :: PackedNeuron -> Neuron
outputNeuron ws = Neuron ws id (const 1)

biasNeuron :: Int -- ^ number of inputs
  -> Neuron
biasNeuron nInputs = Neuron (replicate nInputs 1) (const 1) (const 0)

createLayer :: [PackedNeuron] -> NeuronCreator -> ALayer ()
createLayer pns nc = map (\pn -> (nc pn,())) pns

sigmoidLayer :: [PackedNeuron] -> ALayer ()
sigmoidLayer pns = (biasNeuron nInputs, ()) : createLayer pns sigmoidNeuron
  where
    nInputs = length $ head pns


outputLayer :: [PackedNeuron] -> ALayer ()
outputLayer pns = createLayer pns outputNeuron -- no need for bias neuron at output layer

createRandomNetwork ::
  Int -- ^ seed for random weigth generator
  -> [Int] -- ^ number of neurons per layer -- [1,1]
  -> Network
createRandomNetwork seed layerNeuronCounts =
    unpackNetwork wss
  where
    restLayerNeuronCounts' = init layerNeuronCounts -- [1]
    hiddenIcsNcs = zip (map (+1) restLayerNeuronCounts') (tail restLayerNeuronCounts') -- :: [(InputCount,NeuronCount)]
--          []                                 [2]                         []
    (outputIc,outputNc) = ((snd $ last hiddenIcsNcs) + 1,last layerNeuronCounts)  -- :: (InputCount,NeuronCount)
    rs = randomRs (-1,1) $ mkStdGen seed
    (hiddenWss,rs') = foldl (\(wss',rs') (ic,nc) -> let
                                (sl,rs'') = icNcToPackedNeurons ic nc rs'
                                in
                               (wss'++[sl],rs'')) ([],rs) hiddenIcsNcs
    (outputWss,_) = icNcToPackedNeurons outputIc outputNc rs'
    wss = hiddenWss ++ [outputWss]

-- ** serialisation deserialization

icNcToPackedNeurons :: Int -> Int -> [Double] -> ([PackedNeuron],[Double])
icNcToPackedNeurons ic nc ws = (take nc $ splitEvery ic ws, drop (ic * nc) ws)

unpackNetwork :: [[PackedNeuron]] -> Network
unpackNetwork wss =
    hLayers ++ [oLayer]
  where
    hLayers = map sigmoidLayer $ init wss
    oLayer = outputLayer $ last wss

packNetwork :: Network -> [[PackedNeuron]]
packNetwork n = (map unpackHiddenLayer (init n)) ++ [unpackLayer (last n)]
  where
    unpackLayer ol = map (ws . fst) ol
    unpackHiddenLayer l = unpackLayer $ tail l -- drop bias neuron


-- * backpropagation algorithm
-- ** forward pass
-- |generate forward pass info for a network
-----------------------------------------------------------
passForward :: Network -> [Double] -> ANetwork ForwardPassInfo
passForward nw xs = reverse $ fst $ foldl pf ([],(1 : xs)) nw -- Die 1 ist der virtuelle BiasInput
  where
    pf (nw',xs') l = (l' : nw', xs'')
      where
        l' = (passForward' l xs')
        xs'' = map (o . snd) l'

-- |generate forward pass info for a layer
-----------------------------------------------------------
passForward' :: ALayer a -> [Double] -> ALayer ForwardPassInfo
passForward' l xs = (map (\(n,_) -> (n, passForward'' n xs)) l)

-- |generate forward pass info for a neuron
-----------------------------------------------------------
passForward'' :: Neuron -> [Double] -> ForwardPassInfo
passForward'' n xs = FPInfo {
    o = (fun n) net',
    net = net',
    xs = xs
  }
  where
    net' = calcNet xs (ws n)

-- |calculate the weigtet input of the neuron
-----------------------------------------------------------
calcNet :: [Double] -> [Double] -> Double
calcNet xs ws = sum $ zipWith (*) xs ws

-- ** weight update
-- |updates the weigts for an entire network
-----------------------------------------------------------
weightUpdate ::
  Double -- ^ learning rate 'alpha'
  -> ANetwork ForwardPassInfo
  -> [Double] -- ^ desired output value
  -> Network
weightUpdate alpha fpnw ys = fst $ foldr (weightUpdate' alpha) ([],ds) fpnw
  where
    ds = zipWith (-) ys (map (o . snd) (last fpnw))

-- |updates the weigts for a layer
-----------------------------------------------------------
weightUpdate' :: Double -> ALayer ForwardPassInfo -> (Network,[Double]) -> (Network,[Double])
weightUpdate' alpha fpl (nw,ds) = (l':nw, ds')
  where
    (l,δs) = unzip $ zipWith (weightUpdate'' alpha) fpl ds
    ds' = map sum $ transpose $ map (\(n,δ) -> map (\w -> w * δ) (ws n)) (zip l δs)
    l' = (map (\n -> (n,())) l)

-- |updates the weigts for a neuron
-----------------------------------------------------------
weightUpdate'' :: Double -> (Neuron, ForwardPassInfo) -> Double -> (Neuron, Double)
weightUpdate'' alpha (n,fpi) d = (n{ws=ws'},δ)
  where
    δ = ((fun' n) (net fpi)) * d
    ws' = zipWith (\x w -> w + (alpha * δ * x)) (xs fpi) (ws n)

-- ** forward pass and weigtupdate put together
backprop ::
  Double -- ^ learning rate 'alpha'
  -> Network
  -> ([Double],[Double]) -- ^ inpit and desired output
  -> Network
backprop alpha nw (xs,ys) = weightUpdate alpha (passForward nw xs) ys

-- * Evaluation
-- |calculates the output of a network for a given input vector
-----------------------------------------------------------
calculate :: Network -> [Double] -> [Double]
calculate nw xs = foldl calculate' (1 : xs) nw -- Die 1 ist der virtuelle BiasInput

-- |calculates the output of a layer for a given input vector
-----------------------------------------------------------
calculate' :: [Double] -> ALayer a -> [Double]
calculate' xs l = map (\(n,_) -> (fun n) (calcNet xs (ws n))) l

-- * Training
-- |quadratic error for a single vector pair
-----------------------------------------------------------
quadErrorNet :: Network -> ([Double], [Double]) -> Double
quadErrorNet nw (xs,ys) = sum $ zipWith (\o y -> (y - o) ** 2) os ys
  where
  os = calculate nw xs

-- |quadratic error for for multiple pairs
-----------------------------------------------------------
globalQuadError :: Network -> [([Double], [Double])] -> Double
globalQuadError nw samples = sum $ map (quadErrorNet nw) samples

-- |produces an indefinite sequence of networks
-----------------------------------------------------------
trainAlot :: 
  Int
  -> Double -- ^ learning rate 'alpha'
  -> Network
  -> [([Double],[Double])] -- ^ list of pairs of input and desired output
  -> [Network]
trainAlot limit alpha nw samples = take limit $
  iterate (\nw' -> foldl (backprop alpha) nw' samples) nw

-- |trains a network with a set of vector pairs until a the 'globalQuadError' is smaller than epsilon
-----------------------------------------------------------
train ::
  Int
  -> Double -- ^ learning rate 'alpha'
  -> Double -- ^ the maximum error 'epsilon'
  -> Network
  -> [([Double],[Double])] -- ^ list of pairs of input and desired output
  -> Maybe Network
train limit alpha epsilon nw samples = find
  (\nw' -> globalQuadError nw' samples < epsilon)
  (trainAlot limit alpha nw samples)

-- tests
testBoolAnd = train 1000 0.5 0.001 (createRandomNetwork 1 [2,2,1])
  [([0,0],[0]),([0,1],[0]),([1,0],[0]),([1,1],[1])]

testBoolOr = train 1000 0.5 0.001 (createRandomNetwork 1 [2,2,1])
  [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[1])]

testBoolXor = train 1000 0.5 0.001 (createRandomNetwork 1 [2,2,1])
  [([0,0],[0]),([0,1],[1]),([1,0],[1]),([1,1],[0])]

testBoolNot = train 1000 0.5 0.001 (createRandomNetwork 1 [1,1,1])
  [([0],[1]),([1],[0])]

