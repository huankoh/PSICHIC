
# def read_molecule
# # Deserialize molecule from the pickle file
# with open('molecule.pkl', 'rb') as f:
#     m2 = pickle.load(f)

# # Print the properties to verify they are preserved
# print('Molecule-level properties:', m2.GetPropsAsDict())
# print('Atom-level properties for the first atom:', m2.GetAtomWithIdx(0).GetPropsAsDict())