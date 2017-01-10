# set of GP elements
# set has variables for:
#   -wav = [SquareTable()} = ?
#   -env = {CosTable, SinTable} = {number of values} = {random.randint(0, 10000), random.float(0, 1)}
#   -met = Metro = {float, float}
#   -amp = TrigEnv() = {}
#   -pit = TrigXnoiseMidi =
#   -out = Osc = ?

#   number of variables total = say, 15 for example
#   __ int, __ float
# # s = Server().boot()
# # s.start()
# # wav = SquareTable()
# # env = CosTable([(0,0), (100,1), (500,.3), (8191,0)])
# # met = Metro(.125, 100).play()
# # amp = TrigEnv(met, table=env, mul=.1)
# # pit = TrigXnoiseMidi(met, dist='loopseg', x1=20, scale=1, mrange=(48,84))
# # out = Osc(table=wav, freq=pit, mul=amp).out()