import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
model_fpath = "/data1/qxwang/checkpoints/simcse-large-gsm8k"
tokenizer = AutoTokenizer.from_pretrained(model_fpath)
model = AutoModel.from_pretrained(model_fpath).cuda()
def compute_embeds_fn(texts, model, tokenizer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().numpy()

ls =[
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
eb = compute_embeds_fn(ls, model, tokenizer)
print(eb[0], eb.shape)