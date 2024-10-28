import gmsh

# Initialize the GMSH API
gmsh.initialize()

# Load the GMSH file
file_path = 'tdcs_simu/ernie_TDCS_1_scalar.msh'
gmsh.open(file_path)

# Get the node and element data
node_tags, node_coords, node_data = gmsh.model.mesh.getNodes()


exit()

element_tags, element_types, element_connectivity = gmsh.model.mesh.getElements()

print(f"node tags: {node_tags}")
print(f"Number of nodes: {len(node_coords) // 3}")
print(f"Number of elements: {len(element_connectivity)}")

exit()

# Optionally, print node and element data
# print("Node coordinates:")
# for i in range(0, len(node_coords), 3):
#     print(f"Node {i // 3}: ({node_coords[i]}, {node_coords[i+1]}, {node_coords[i+2]})")

print("Elements:")
for i in range(len(element_tags)):
    print(f"Element {i}: Type={element_types[i]}, Connectivity={element_connectivity[i]}")

# Finalize the GMSH API
gmsh.finalize()
