module AI.BPANN.Async where

import Data.List
import Data.List.Split
import Data.Maybe
import System.Random
import AI.BPANN
import Control.Concurrent.Async
import Data.Foldable

createLayerIO :: [PackedNeuron] -> NeuronCreator -> IO (ALayer ())
createLayerIO pns nc = mapConcurrently (\pn -> return (nc pn,())) pns

sigmoidLayerIO :: [PackedNeuron] -> IO (ALayer ())
sigmoidLayerIO pns = do
	l <- createLayerIO pns sigmoidNeuron
	return $ (biasNeuron nInputs, ()) : l
  where
    nInputs = length $ head pns

outputLayerIO :: [PackedNeuron] -> IO (ALayer ())
outputLayerIO pns = createLayerIO pns outputNeuron

unpackNetworkIO :: [[PackedNeuron]] -> IO Network
unpackNetworkIO wss = do
	hLayers <- mapConcurrently (return . sigmoidLayer) $ init wss
	return $ hLayers ++ [oLayer]
  where
    oLayer = outputLayer $ last wss

createRandomNetworkIO ::
  Int -- ^ seed for random weigth generator
  -> [Int] -- ^ number of neurons per layer -- [1,1]
  -> IO Network
createRandomNetworkIO seed layerNeuronCounts =
    unpackNetworkIO wss
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

packNetworkIO :: Network -> IO [[PackedNeuron]]
packNetworkIO n = do
	l <- mapConcurrently unpackHiddenLayer (init n)
	l2 <- unpackLayer (last n)
	return $ l ++ [l2]
  where
    unpackLayer ol = mapConcurrently (return . ws . fst) ol
    unpackHiddenLayer l = unpackLayer $ tail l

passForwardIO' :: ALayer a -> [Double] -> IO (ALayer ForwardPassInfo)
passForwardIO' l xs = (mapConcurrently (\(n,_) -> return (n, passForward'' n xs)) l)

passForwardIO :: Network -> [Double] -> IO (ANetwork ForwardPassInfo)
passForwardIO nw xs = fmap reverse $ fmap fst $ foldlM pf ([],(1 : xs)) nw -- Die 1 ist der virtuelle BiasInput
  where
    pf (nw',xs') l = do
		l' <- passForwardIO' l xs'
		xs'' <- mapConcurrently (return . o . snd) l'
		return (l' : nw', xs'')

weightUpdateIO'' :: Double -> (Neuron, ForwardPassInfo) -> Double -> IO (Neuron, Double)
weightUpdateIO'' alpha (n,fpi) d = do
	ws' <- mapConcurrently (\(x,w) -> return $ w + (alpha * δ * x)) $ zip (xs fpi) (ws n)
	return (n{ws=ws'},δ)
  where
    δ = ((fun' n) (net fpi)) * d
    
weightUpdateIO' :: Double -> ALayer ForwardPassInfo -> (Network,[Double]) -> IO (Network,[Double])
weightUpdateIO' alpha fpl (nw,ds) = do
	(l,δs) <- fmap unzip $ mapConcurrently (\(x,y)-> weightUpdateIO'' alpha x y) $ zip fpl ds
	ds' <- (fmap . fmap) sum $ fmap transpose $
		mapConcurrently (\(n,δ) -> mapConcurrently (\w -> return $ w * δ) (ws n)) (zip l δs)
	let l' = (map (\n -> (n,())) l)
	return (l':nw, ds')

weightUpdateIO ::
  Double -- ^ learning rate 'alpha'
  -> ANetwork ForwardPassInfo
  -> [Double] -- ^ desired output value
  -> IO Network
weightUpdateIO alpha fpnw ys = do
	ds <- mapConcurrently (\(x,y)-> return $ x - y) $ zip ys (map (o . snd) (last fpnw))
	fmap fst $ foldrM (weightUpdateIO' alpha) ([],ds) fpnw
    
backpropIO ::
  Double -- ^ learning rate 'alpha'
  -> Network
  -> ([Double],[Double]) -- ^ inpit and desired output
  -> IO Network
backpropIO alpha nw (xs,ys) = do
	x <- passForwardIO nw xs
	weightUpdateIO alpha x ys

calculateIO' :: [Double] -> ALayer a -> IO [Double]
calculateIO' xs l = mapConcurrently (\(n,_) -> return $ (fun n) (calcNet xs (ws n))) l

calculateIO :: Network -> [Double] -> IO [Double]
calculateIO nw xs = foldlM calculateIO' (1 : xs) nw

quadErrorNetIO :: Network -> ([Double], [Double]) -> IO Double
quadErrorNetIO nw (xs,ys) = do 
	os <- calculateIO nw xs
	fmap sum $ mapConcurrently (\(o, y) -> return $ (y - o) ** 2) $ zip os ys

globalQuadErrorIO :: Network -> [([Double], [Double])] -> IO Double
globalQuadErrorIO nw samples = fmap sum $ mapConcurrently (quadErrorNetIO nw) samples

trainAlotIO :: 
  Int
  -> Double -- ^ learning rate 'alpha'
  -> Network
  -> [([Double],[Double])] -- ^ list of pairs of input and desired output
  -> IO [Network]
trainAlotIO limit alpha nw samples = fmap (take limit) $ sequence $
  iterate (>>= (\nw' -> foldlM (backpropIO alpha) nw' samples)) (return nw)

trainIO ::
  Int
  -> Double -- ^ learning rate 'alpha'
  -> Double -- ^ the maximum error 'epsilon'
  -> Network
  -> [([Double],[Double])] -- ^ list of pairs of input and desired output
  -> IO (Maybe Network)
trainIO limit alpha epsilon nw samples = do 
	ta <- trainAlotIO limit alpha nw samples
	ta2 <- mapConcurrently (\nw' -> do
		q <- globalQuadErrorIO nw' samples
		return (q < epsilon,nw') ) $ ta
	return $ fmap snd $ find (\(x,_)->x) ta2