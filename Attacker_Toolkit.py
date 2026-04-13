from langchain.tools import tool
import random

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method, AdversarialPatch

from art.attacks.evasion import CarliniWagnerASR, ImperceptibleASR, ImperceptibleASRPyTorch, BoundaryAttack

from art.attacks.evasion import ZooAttack, DecisionTreeAttack
@tool
def vision_attack_toolkit(attack_type: str, epsilon: float, target_model: str, ):
    """
    This is the Vision attack toolkit, which will execute a specific vision attack using the ART library. 
    - attack_type: Choose from ['fgsm', 'pgd', 'deepfool', 'calinil2', 'adversarial_patch']
    - epsilon: Perturbation strength (eg. 0.1)
    - target_model: The model to be tested on (eg. 'resnet50', 'vgg16' etc.)
    """

    x_test = 0.0 # get loaded data based on the model

    if attack_type == 'fgsm':
        attack = FastGradientMethod(estimator = target_model, eps = epsilon)
    elif attack_type == 'pgd': 
        attack = ProjectedGradientDescent(estimator = target_model, eps = epsilon)
    elif attack_type == 'deepfool':
        attack = DeepFool(estimator = target_model)
    elif attack_type == 'calinil2':
        attack = CarliniL2Method(estimator = target_model, confidence = epsilon)
    elif attack_type == 'adversarial_patch':
        attack = AdversarialPatch(estimator = target_model, patch_shape=(3,224,224), eps = epsilon)
    else: 
        return "Unsupported attack type. Choose from ['fgsm', 'pgd', 'deepfool', 'calinil2', 'adversarial_patch']"
    
    x_adv = attack.generate(x = x_test)

    return {'status': 'success', 'adversarial_example': x_adv, 'metrics':{'epsilon': epsilon, 'attack_type': attack_type}}


@tool
def audio_attack_toolkit(attack_type: str, epsilon: float, target_model: str):
    """
    This is the Audio attack toolkit, which will execute a specific audio attack using the ART library. 
    - attack_type: Choose from ['cw_asr','imperceptible_asr', 'imperceptible_pytorch', boundary_attack']
    - epsilon: Perturbation strength (eg. 0.1)
    - target_model: The model to be tested on (eg. 'resnet50', 'vgg16' etc.)
    """

    x_test = 0.0 # need to get data

    if attack_type == 'cw_asr':
        attack = CarliniWagnerASR(estimator = target_model, learning_rate = 0.1, max_iter = 100, confidence = epsilon)
    elif attack_type == 'imperceptible_asr':
        attack= ImperceptibleASR(estimator = target_model, eps = epsilon, max_iter = 100)
    elif attack_type == 'imperceptible_pytorch':
        attack = ImperceptibleASRPyTorch(estimator = target_model, eps = epsilon, max_iter = 100)
    elif attack_type == 'boundary_attack':
        attack = BoundaryAttack(estimator = target_model, targeted = False, max_iter = 500, delta = 0.01, epsilon = epsilon)
    else: 
        return "Unsupported attack type. Choose from ['cw_asr','imperceptible_asr', 'imperceptible_pytorch', 'boundary_attack']"
    
    x_adv = attack.generate(x = x_test)

    return {'status': 'success', 'adversarial_example': x_adv, 'metrics':{'epsilon': epsilon, 'attack_type': attack_type}}



@tool
def structuredData_attack_toolkit(attack_type: str, epsilon: float, target_model: str):
    """
    This is the Structured Data attack toolkit, which will execute a specific Structured Data attack using the ART library. 
    - attack_type: Choose from ['zoo','boundary_attack','decision_tree']
    - epsilon: Perturbation strength (eg. 0.1)
    - target_model: The model to be tested on (eg. 'resnet50', 'vgg16' etc.)
    """

    x_test = 0.0 # need to get data

    if attack_type == 'zoo':
        attack = ZooAttack(estimator = target_model, max_iter = 100, confidence = epsilon)
    elif attack_type == 'boundary_attack':
        attack= BoundaryAttack(estimator = target_model, targeted = False, max_iter = 100, epsilon = epsilon)
    elif attack_type == 'decision_tree':
        attack = DecisionTreeAttack(classifier = target_model)
    else: 
        return "Unsupported attack type. Choose from ['zoo','boundary_attack','decision_tree']"
    
    x_adv = attack.generate(x = x_test)

    return {'status': 'success', 'adversarial_example': x_adv, 'metrics':{'epsilon': epsilon, 'attack_type': attack_type}}




@tool
def LLM_attack_toolkit(attack_strategy: str, original_prompt: str, target_model: str):
    """ 
    Toolkit for LLM-based text attacks. (Jailbreaking, Prompt Injection, etc.)
    - attack_strategy: The strategy decided by the Planner (eg. 'jailbreak', 'obfuscation')
    - original_prompt: Malicious intent to be disguised. 
    - target_model: The LLM being attacked (eg. 'Llama-3')
    """

    if 'jailbreak' in attack_strategy.lower():
        payload = f'System Override: {attack_strategy}] {original_prompt}'
    elif 'obfuscation' in attack_strategy.lower():
        payload = ''.join(random.sample(original_prompt, len(original_prompt))) # simple obfuscation by shuffling characters
    else: 
        payload = f'{attack_strategy} {original_prompt}'

    return {'status': 'payload_generated', 'attack_payload': payload, 'target': target_model}


