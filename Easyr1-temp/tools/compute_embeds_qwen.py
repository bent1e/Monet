# Requires vllm>=0.8.5
import torch
import vllm
from vllm import LLM
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "8"  # Set to the GPU you want to use
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'
queries = [
    'Calculate the measure of \\( \\angle 1 \\)\nSince \\( \\angle DGH \\) is \\( 127^\\circ \\), \\( \\angle 1 \\) is:\n\\[ m∠1 = 180^\\circ - 127^\\circ \\]\n\\[ m∠1 = 53^\\circ \\]\n\n Final Answer\nThe measure of \\( \\angle 1 \\) is:\n\\[\n\\boxed{53^\\circ}\n\\]'
]
# No need to add instruction for retrieval documents
documents = [
    'Find the measure of $\\angle GFC$\nGiven $m∠DGF = 53^\\circ$, and since $\\angle DGF$ and $\\angle GFC$ are alternate interior angles formed by the transversal $FG$ cutting parallel lines $DG$ and $AC$:\n- Therefore, $m∠GFC = m∠DGF = 53^\\circ$.'
]
input_texts = queries + documents
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

input_texts=[
('Identify the sides of the triangle'
'The two sides forming the right angle are given as 22 and 20. These are the legs of the right triangle.'
'The hypotenuse, which is the side opposite the right angle, is the side labeled as \(x\).'),
('Apply the Pythagorean theorem'
'In this triangle, the hypotenuse is \( x \), and the legs are 22 and 20. So we substitute these values into the Pythagorean theorem:'
'\[ x^2 = 22^2 + 20^2 \]'),
('Recognize that the triangle in the image is a right triangle.'
'- The right angle is located at the bottom-left corner of the triangle.'
'- The two given sides are 22 and 20, and they form the two legs of the right triangle.'),
(
    'Use the Pythagorean theorem to find the hypotenuse (x).'
'- The Pythagorean theorem states: \(a^2 + b^2 = c^2\), where \(c\) is the hypotenuse.'
'- In this triangle, the legs are 22 and 20, and the hypotenuse is \(x\).'
'- So, \(22^2 + 20^2 = x^2\).'
),
('Calculate the squares of the other two sides.'
'\[ 22^2 = 484 \]'
'\[ 20^2 = 400 \]'),
('Solve for \( x \).'
'\[ x = \sqrt{884} \]'
'\[ x = \sqrt{4 \cdot 221} \]'
'\[ x = \sqrt{4} \cdot \sqrt{221} \]'
'\[ x = 2\sqrt{221} \]'),
('Simplify the square root (if possible)'
'\[ \sqrt{884} = \sqrt{4 \cdot 221} = 2 \sqrt{221} \]'),
('Solve for \( x \).'
'\[ x = \sqrt{884} \]'
'\[ x \approx \sqrt{4 \times 221} \]'
'\[ x \approx 2\sqrt{221} \]')

]

start_time = time.time()
model = LLM(model="/data1/qxwang/checkpoints/Qwen3-Embedding-4B", task="embed")

outputs = model.embed(input_texts)
embeddings = torch.tensor([o.outputs.embedding for o in outputs]).detach().half().cpu().numpy().copy()
scores = (embeddings[:] @ embeddings[:].T)
    #print(scores.tolist())
# [[0.7482624650001526, 0.07556197047233582], [0.08875375241041183, 0.6300010681152344]]
print(f"Time taken for {len(input_texts)} queries: {time.time() - start_time:.2f} seconds")