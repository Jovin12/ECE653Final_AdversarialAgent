from langchain.tools import tool
import random
import numpy as np

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method, AdversarialPatch

from art.attacks.evasion import CarliniWagnerASR, ImperceptibleASR, ImperceptibleASRPyTorch, BoundaryAttack

from art.attacks.evasion import ZooAttack, DecisionTreeAttack


# --------------
# Vision Attacks
# --------------
@tool
def vision_attack_toolkit(attack_type: str, epsilon: float, art_classifier, x_test: np.ndarray):
    """
    Executes a vision adversarial attack using the ART library.
 
    Args:
        attack_type   : One of ['fgsm', 'pgd', 'deepfool', 'carlini_l2', 'adversarial_patch']
        epsilon       : Perturbation strength (e.g. 0.03). Used as confidence for carlini_l2.
        art_classifier: A PyTorchClassifier instance (from vision_target.py). NOT a string.
        x_test        : Clean input images as np.ndarray of shape (N, 3, 224, 224), float32 [0,1].
 
    Returns:
        dict with keys: status, adversarial_examples, original_examples, metrics
    """
    attack_type = attack_type.lower()

    if attack_type == 'fgsm':
        attack = FastGradientMethod(estimator = art_classifier, eps = epsilon)
    elif attack_type == 'pgd': 
        attack = ProjectedGradientDescent(estimator = art_classifier, eps = epsilon, eps_step = epsilon/4, max_iter=40)
    elif attack_type == 'deepfool':
        attack = DeepFool(classifier = art_classifier, max_iter=10)
    elif attack_type == 'calinil2':
        attack = CarliniL2Method(estimator = art_classifier, confidence = epsilon, max_iter=100)
    elif attack_type == 'adversarial_patch':
        attack = AdversarialPatch(classifier = art_classifier, patch_shape=(3,64,64), 
                                  rotation_max=22.5, scale_min=0.1, scale_max=1.0, max_iter=500, batch_size=100)
    else: 
        return {
            "status": "error",
            "message": f"Unsupported attack: '{attack_type}'. "
                       f"Choose from ['fgsm', 'pgd', 'deepfool', 'carlini_l2', 'adversarial_patch']"
        }
    
    x_adv = attack.generate(x = x_test)

    return {
        'status': 'success', 
        'adversarial_example': x_adv, 
        'metrics':{'epsilon': epsilon, 
        'attack_type': attack_type}, 
        'num_samples': len(x_test)
            }


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


