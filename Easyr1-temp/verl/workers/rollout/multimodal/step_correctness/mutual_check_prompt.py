
######################################
# mask
######################################

sys_prompt_mask_accum_step = (f"You are given a question text $Q$ of a vision-language reasoning task, and a reasoning trajectory $S$ that contains several reasoning steps $s_i$, $i=1,...,T$." 
"Please analyze $Q$ and $S$ to determine whether the last step $s_T$ includes any observation of the image or reasoning results."
"\nIf $s_T$ does not include any perception of the image or reasoning contents, output 'no';"
"\nIf $s_T$ does include any perception of the image or reasoning contents, replace the observed information or derived results with '<blank>' while leaving the rest of the text in $s_T$ exactly as it is, then output the modified last step $s_T$."
"\nDon't output any other text, code, or explanations."
"\nFor example:"
"\nQuestion $Q$: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: When the ant <image1> walks from home <image2> along the arrows $\\rightarrow 3, \\uparrow 3, \\rightarrow 3, \\uparrow 1$, he gets to the ladybird <image3>."
"\nWhich animal does the ant <image1> get to when he walks from home <image2> along the following arrows: $\\rightarrow 2, \\downarrow 2, \\rightarrow 3, \\uparrow 3, \\rightarrow 2, \\uparrow 2$?\n<image6>\n<image7\nChoices:\n(A) A\n(B) B\n(C) C\n(D) D\n(E) E"
"\n\nReasoning trajectory $S$:  <image6>:\n\nStep 1. Start at home <image2>."
"\nStep 2. Move right 2 steps: The ant is now at the first empty square to the right."
"\nStep 3. Move down 2 steps: The ant is now at the second column, third row."
"\nStep 4. Move right 3 steps: The ant is now at the fifth column, third row."
"\n\nYour output: Step 4. Move right 3 steps: The ant is now at the <blank> column, <blank> row."
)


sys_prompt_mask_single_step = (f"You are given a question $Q$ of a vision-language reasoning task, and a reasoning step $s$." 
"Please analyze $Q$ and $s$ to determine whether $s$ includes any observation of the image, reasoning results, or analysis of the given question and conditions."
"\nIf $s$ does not include any perception of the image or reasoning contents, output 'no';"
"\nIf $s$ does include any perception of the image or reasoning contents, replace the observed information or derived results with '<blank>' while leaving the rest of the text in $s$ exactly as it is, then output the modified $s$."
"\n\nHere are some examples:"
#"\n\nquestion $Q$: \n\nQuestion:\n\nHint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: What type of Olympics might be held in this area?\nChoices:\n(A) autumn games\n(B) winter games\n(C) spring games\n(D) summer games\n\n"
#"\nReasoning step $s$: - **Autumn Games (Choice A):** These are not specific to any particular season and include sports such as athletics and swimming that are not related to snow.\n- **Winter Games (Choice B):** These are games focused specifically on winter sports and activities. The area shown in the image is a snowy mountain where winter sports like skiing and snowboarding are being practiced.\n- **Spring Games (Choice C):** These would involve sports played in a transitional season and are not typically snow-related.\n- **Summer Games (Choice D):** These include sports played during the warmer seasons and do not involve snow."
#"\nYour output: - **Autumn Games (Choice A):** These are not specific to any particular season and include sports such as athletics and swimming that are not related to snow.\n- **Winter Games (Choice B):** These are games focused specifically on winter sports and activities. The area shown in the image is <blank>.\n- **Spring Games (Choice C):** These would involve sports played in a transitional season and are not typically snow-related.\n- **Summer Games (Choice D):** These include sports played during the warmer seasons and do not involve snow."
"\n\n\nQuestion $Q$: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
"\n\nQuestion: Four of the following five pictures show pieces of the graph of the same quadratic function. Which piece does not belong?"
"\nChoices:"
"\n(A) A"
"\n(B) B"
"\n(C) C"
"\n(D) D"
"\n(E) E"
"\n\nReasoning step $s$: Analysis:"
"\n- **Option A**: A curve increasing from left to right, consistent with a quadratic function opening upwards."
"\n- **Option B**: A curve decreasing from left to right, consistent with a quadratic function opening downwards."
"\n- **Option C**: A line, not a curve, inconsistent with a quadratic function."
"\n\nYour output: Analysis:"
"\n- **Option A**: A curve increasing from <blank> to <blank>, consistent with a <blank> function opening <blank>."
"\n- **Option B**: A curve decreasing from <blank> to <blank>, consistent with a <blank> function opening downwards."
"\n- **Option C**: A <blank>, not a <blank>, inconsistent with a <blank> function."
"\n\n\nQuestion $Q$: When a spring does work on an object, we cannot find the work by simply multiplying the spring force by the object's displacement. The reason is that there is no one value for the force-it changes. However, we can split the displacement up into an infinite number of tiny parts and then approximate the force in each as being constant. Integration sums the work done in all those parts. Here we use the generic result of the integration.\r\n\r\nIn Figure, a cumin canister of mass $m=0.40 \\mathrm{~kg}$ slides across a horizontal frictionless counter with speed $v=0.50 \\mathrm{~m} / \\mathrm{s}$. It then runs into and compresses a spring of spring constant $k=750 \\mathrm{~N} / \\mathrm{m}$. When the canister is momentarily stopped by the spring, by what distance $d$ is the spring compressed?"
"\n\nReasoning step $s$: Step 2: The work done by the spring force is given by $W = \\int_{0}^{d} F \\, dx = \\int_{0}^{d} -kx \\, dx = -\\frac{1}{2}kx^2$ evaluated from $0$ to $d$, which is $W = -\\frac{1}{2}kd^2$."
"\n\nYour output: Step 2: The work done by the spring force is given by $W = <blank>$ evaluated from $0$ to $d$, which is $W = <blank>$."
"\n\n\nquestion $Q$: What type of Olympics might be held in this area?\n(A) autumn games\n(B) winter games\n(C) spring games\n(D) summer games"
"\n\nReasoning step $s$: Next, we need to understand what types of Olympic Games are available."
"\n\nYour output: no"
"\n\n\nYou must strictly follow the instruction: **Only output the masked reasoning step or 'no', and immediately stop outputting after that. Do not output anything else, including any reasoning, explanation, analysis, code, or comments.**"
)

sys_prompt_mask_single_step_no_question = (f"You are given some Reasoning stepS $s$ of a vision-language reasoning task." 
"Please analyze $s$ to determine whether it includes any observation of the image, reasoning results, or analysis of the given question and conditions."
"\nIf $s$ does not include any perception of the image or reasoning contents, output 'no';"
"\nIf $s$ does include any perception of the image or reasoning contents, replace the observed information or derived results with '<blank>' while leaving the rest of the text in $s$ exactly as it is, then output the modified $s$."
"\nYou are only allowed to output the masked step or 'no', don't answer the question or output new reasoning contents."
"\n\nHere are some examples:"
"\n\nReasoning step $s$: Step by step reasoning:"
"\n1. The image showcases a winding road in a mountainous area."
"\n2. Mountainous areas are prone to landslides due to the erosion of soil and rock, which can be caused by factors such as heavy rainfall, earthquakes, or volcanic activity."
"\n3. Floods and lightning are less likely to obstruct movement on a road in this specific context because the image suggests a dry environment, and while lightning could be a concern, it does not seem to have an immediate or direct impact on the road."
"\n4. Landslides are known to be common in mountainous regions and can occur due to natural events without significant trigger mechanisms, unlike earthquakes."
"\n\nYOUR OUTPUI $s$: Step by step reasoning:"
"\n1. The image showcases <blank>"
"\n2. Mountainous areas are prone to landslides due to the erosion of soil and rock, which can be caused by factors such as heavy rainfall, earthquakes, or volcanic activity."
"\n3. Floods and lightning are less likely to obstruct movement on a road in this specific context because the image suggests <blank>."
"\n4. Landslides are known to be common in mountainous regions and can occur due to natural events without significant trigger mechanisms, unlike earthquakes."
"\n\nReasoning step $s$: Step 2: The work done by the spring force is given by $W = \\int_{0}^{d} F \\, dx = \\int_{0}^{d} -kx \\, dx = -\\frac{1}{2}kx^2$ evaluated from $0$ to $d$, which is $W = -\\frac{1}{2}kd^2$."
"\nYour output: Step 2: The work done by the spring force is given by $W = <blank>$ evaluated from $0$ to $d$, which is $W = <blank>$."
"\n\nReasoning stepS $s$: Next, we need to understand what types of Olympic Games are available."
"\nYour output: no"
)





sys_prompt_mask_consec_step = (f"You are given a question text $Q$ of a vision-language reasoning task, and two consective text reasoning steps $i-1$ and $i$." 
"Please analyze $Q$, step $i-1$, and step $i$ to determine whether step $i$ includes any observation of the image or reasoning results."
"\nIf step $i$ does not include any perception of the image or reasoning contents, output 'no';"
"\nIf step $i$ does include any perception of the image or reasoning contents, replace the observed information or derived results with '<blank>' while leaving the rest of the text in step $i$ exactly as it is, then output the modified step $i$. You don't need to output step $i-1$."
"\nDon't output any other text, code, or explanations."
"\nFor example:"
"\nQuestion $Q$: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: When the ant <image1> walks from home <image2> along the arrows $\\rightarrow 3, \\uparrow 3, \\rightarrow 3, \\uparrow 1$, he gets to the ladybird <image3>."
"\nWhich animal does the ant <image1> get to when he walks from home <image2> along the following arrows: $\\rightarrow 2, \\downarrow 2, \\rightarrow 3, \\uparrow 3, \\rightarrow 2, \\uparrow 2$?\n<image6>\n<image7\nChoices:\n(A) A\n(B) B\n(C) C\n(D) D\n(E) E"
"\n\nReasoning step $i-1$: Step 5: Substituting the given values into the equation, we get $d = \\sqrt{\\frac{0.40 \\mathrm{~kg} \\times (0.50 \\mathrm{~m/s})^2}{750 \\mathrm{~N/m}}}$.\n###"
"\nReasoning step $i$: Step 6: Solving the equation, we get $d = \\sqrt{\\frac{0.20 \\mathrm{~kg \\cdot m^2/s^2}}{750 \\mathrm{~N/m}}} = \\sqrt{\\frac{0.20}{750}} = \\sqrt{0.0002667} \\approx 0.0163 \\mathrm{~m}$."
"\n\nYour output: Step 6: Solving the equation, we get $d = \\sqrt{\\frac{0.20 \\mathrm{~kg \\cdot m^2/s^2}}{750 \\mathrm{~N/m}}} = \\sqrt{\\frac{<blank>}{<blank>}} = \\sqrt{<blank>} \\approx <blank> \\mathrm{~m}$."
#"\n\nReasoning step $i-1$: Step 3: The kinetic energy of the canister just before it hits the spring is $K_i = \\frac{1}{2}mv^2$. This kinetic energy is converted into the potential energy of the spring when the canister is stopped. The potential energy of the spring when it is compressed by a distance $d$ is $U = \\frac{1}{2}kd^2$."
#"\nReasoning step $i$: Step 4: Setting the initial kinetic energy equal to the potential energy of the spring, we get:\n$$\\frac{1}{2}mv^2 = \\frac{1}{2}kd^2$$"
#"\n\nYour output: Step 4: Setting the initial kinetic energy equal to the potential energy of the spring, we get: <blank>."
)



######################################
# fill
######################################

sys_prompt_fill_single_step = (f"You are given an IMAGE $I$, a question $Q$, and a Reasoning step $s_i$." 
"In $s_i$, the crucial information is masked with '<blank>'."
"Please analyze the IMAGE, the question, and the the unmasked information in Reasoning step $s_i$ to infer the correct contents masked by '<blank>'." 
"Output your modified step $i$ with '<blank>' filled by the correct contents while leaving the rest of the text in $s_i$ exactly as it is."
"Don't output the reasoning steps after STEP $i$."
"\n\nFor example:"
"\nIMAGE $I$: <image>"
"\nquestion $Q$: Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all small gray spheres. Subtract all cylinders. How many objects are left?"
"\nReasoning step $s_i$: Step 2: Subtract all small gray spheres. There is <blank> small gray sphere, so we subtract <blank>."
"\nYour output: Step 2: Subtract all small gray spheres. There is one small gray sphere, so we subtract 1."           
)

sys_prompt_fill_consec_step = (f"You are given an IMAGE $I$, a question $Q$, and two consecutive Reasoning stepS $i-1$ and $i$." 
"In the latter step $i$, the crucial information is masked with '<blank>'."
"Please analyze the IMAGE, the question, and the unmasked information in the Reasoning stepS to infer the correct contents masked by '<blank>'." 
"Output your modified Reasoning step $i$ with '<blank>' filled by the correct contents while leaving the rest of the text in step $i$ exactly as it is."
"Don't output the reasoning steps after STEP $i$."
"\n\nFor example:"
"\nIMAGE $I$: <image>"
"\nquestion $Q$: Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all small gray spheres. Subtract all cylinders. How many objects are left?"
"\nReasoning step $i-1$: Step 1: Start with the total number of objects in the image. There are 4 objects: a red cylinder, a red sphere, and two gray spheres, a big one and a small one."
"\nReasoning step $i$: Step 2: Subtract all small gray spheres. There is <blank> small gray sphere, so we subtract <blank>."
"\nYour output: Step 2: Subtract all small gray spheres. There is one small gray sphere, so we subtract 1."           
)


sys_prompt_fill_accum_step = (f"You are given an image, a question, and a reasoning trajectory $S$ that contains several reasoning steps $s_i$, $i=1,...,T$." 
" In the last step $s_T$, the crucial information is masked with '<blank>'."
" Please analyze the image, the question, the reasoning trajectory and the unmasked information in $s_T$ to infer the correct contents masked by '<blank>'." 
" Output your modified last step $s_T$ with '<blank>' filled by the correct contents while leaving the rest of the text in $s_T$ exactly as it is."
"\nRemember:"
"\n1. You are only allow to output the modified last step $s_T$."
"\n2. Don't output any other text, code, or explanations." 
"\n3. Don't repeat the reasoning trajectories before the last step $s_T$."
"\n4. Don't output the reasoning steps after STEP $s_T$."
"\n\nFor example:"
"\nImage $I$: <image>"
"\nQuestion $Q$: Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all small gray spheres. Subtract all cylinders. How many objects are left?"
"\n\nReasoning trajectory $S$:"
"\nStep 1: Start with the total number of objects in the image. There are 4 objects: a red cylinder, a red sphere, and two gray spheres."
"\nStep 2: Subtract all small gray spheres. There are two small gray spheres, so we subtract 2."
"\nStep 3: Subtract all cylinders. There is 1 red cylinder in the image."
"\nLast step $s_T$: Step 4: Calculate the remaining objects: <blank> total objects - <blank> small gray spheres - <blank> cylinder = <blank> object."
"\nYour output: Step 4: Calculate the remaining objects: 4 total objects - 2 small gray spheres - 1 cylinder = 1 object."           
)


sys_prompt_fill_accum_step_mask_random = (f"You are given an image, a question, and a reasoning trajectory $S$ that contains several reasoning steps $s_i$, $i=1,...,T$." 
"In the last step $s_T$, the words of crucial information is masked with '<blank>'."
"Please analyze the image, the question, the reasoning trajectory and the unmasked information in $s_T$ to infer the correct contents masked by '<blank>'." 
"Output your modified last step $s_T$ with '<blank>' filled by the correct contents while leaving the rest of the text in $s_T$ exactly as it is."
"\nDon't output any other text, code, or explanations."
"\n\nFor example:"
"\nImage $I$: <image>"
"\nQuestion $Q$: Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Subtract all small gray spheres. Subtract all cylinders. How many objects are left?"
"\n\nReasoning trajectory $S$:"
"\nStep 1: Start with the total number of objects in the image. There are 4 objects: a red cylinder, a red sphere, and two gray spheres."
"\nStep 2: Subtract all small gray spheres. There are two small gray spheres, so we subtract 2."
"\nStep 3: Subtract all cylinders. There is 1 red cylinder in the image."
"\nStep 4: Calculate the remaining objects:  4 total objects - 2 <blank> <blank> - <blank> cylinder = <blank> object."
"\nYour output: Step 4: Calculate the remaining objects: 4 total objects - 2 small gray spheres - 1 cylinder = 1 object."           
)




######################################
# check
######################################


sys_prompt_check_single_step = (
    f"You are given two reasoning steps $s_1$ and $s_2$. Please analyze $s_1$ and $s_2$ to determine whether they are consistent with each other."
    "\nIf $s_1$ and $s_2$ are consistent with each other, output '1';"
    "\nIf $s_1$ and $s_2$ are inconsistent with each other, output '0'."
    "\nOutput your answer as a single digit (0 or 1) without any explanation."
    "\n\nFor example:"
    "\nReasoning step $s_1$: Step 2: Subtract all small gray spheres. There are two small gray spheres, so we subtract 2."
    "\nReasoning step $s_2$: Step 2: Subtract all small gray spheres. There is one small gray sphere, so we subtract 1."
    "\nYour output: 0"
    "\n\nReasoning step $s_1$: Step1: Identify the objects in the image.\n- There are two gray spheres.\n- There is one red cylinder.\n- There is one red sphere.\n- There is one large gray sphere.\n- There is a small gray metallic sphere"
    "\nReasoning step $s_1$: Step1: Identify the objects in the image.\n- There are two gray spheres.\n- There is a red cylinder.\n- There is a red sphere.\n- There is a large gray metallic sphere.\n- There is a small gray metallic sphere"
    "\nYour output: 1"
)

sys_prompt_check_consec_step = (
    f"You are given two reasoning trajectories $S_1$ and $S_2$, each contains two consecutive steps. Please analyze $S_1$ and $S_2$ to determine whether they are consistent with each other."
    "\nIf $S_1$ and $S_2$ are consistent with each other, output '1';"
    "\nIf $S_1$ and $S_2$ are inconsistent with each other, output '0'."
    "\nOutput your answer as a single digit (0 or 1) without any explanation."
    "\n\nFor example:"
    "\nReasoning trajectory $S_1$: "
    "\nReasoning trajectory $S_2$: "
    "\nYour output: 1"
)