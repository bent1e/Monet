# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("/data1/qxwang/checkpoints/Qwen3-Embedding-4B", device="cpu")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-8B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

input_texts=[
    'Calculate the central angle\nSince PQRS is a square inscribed in the circle, each central angle subtended by the sides of the square is 90 degrees because the square divides the circle into four equal parts.',
'Determine the angles in the square.\n- A square has all four angles equal to 90 degrees.\n- Therefore, \\( \\angle QRP = 90^\\circ \\).',
'Determine the angles in the square.\n- A square has all four angles equal to 90 degrees.\n- Therefore, \\( \\angle QRP = 90^\\circ \\).',
'Calculate the angles of the rhombus\nIn a rhombus, all sides are equal, and the diagonals bisect each other at right angles, making the diagonals perpendicular. The diagonals of a rhombus that are diameters of the circle will divide the circle into four equal segments, each subtending a central angle of 90 degrees.\n\nThe diagonals of the rhombus will bisect each other at right angles, and the angle created by the intersection of the diagonals in the center of the circle will be 90 degrees.',
'Analyze the angles in a square\nIn a square, all four angles are 90 degrees. Therefore, angle QRP in the square will be 90 degrees.',
'Conclusion\nSince PQRS is a square inscribed in a circle, and we are asked to find the measure of angle QRP, we conclude that angle QRP is 90 degrees.\n\n\\boxed{90}',
"Calculate the angle \\( \\angle QR P \\)\nSince PQRS is a square (because the rhombus inscribed in a circle is a square, and square's diagonals are equal and bisect each other at right angles), the central angle subtended by each side of the square at the center of the circle is 90 degrees. Therefore, \\( \\angle QR P \\) is 90 degrees because it is a central angle subtended by one of the sides of the square.\n\nThe measure of \\( \\angle Q R P \\) is \\boxed{90} degrees.",
'Calculate angle properties\nSince the rhombus is divided into four isosceles triangles by its diagonals, and the diagonals of the rhombus are diameters of the circle, each angle created by the diagonals will be half of the straight angle (180 degrees) for the rhombus. Therefore, each angle formed at the center by the diagonals is 90 degrees.',
'Calculate the angle of the rhombus\nIn a square (which is a specific type of rhombus), each internal angle is 90 degrees.',
'Find the measure of \\(\\angle QR P\\)\nSince PQRS is a square, \\(\\angle QRP\\) is one of the angles in the rhombus and thus, 90 degrees.\n\nThe final answer is \\(\\boxed{90}\\).'
]
# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
print(query_embeddings)
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7493, 0.0751],
#         [0.0880, 0.6318]])
