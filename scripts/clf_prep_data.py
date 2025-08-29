from neutrino.clf.prepare import TensorPair

pair = TensorPair.load_tensor()

print("=== TensorPair Test ===")
print("A shape:", pair.A.shape)
print("B shape:", pair.B.shape)
print("Both shapes via property:", pair.shapes)
print("Number of columns:", len(pair.columns))
print("First 5 columns:", pair.columns[:5])
