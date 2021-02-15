from typing import Union, Any
import torch as th


def tensor_dict_eq(dict1: dict, dict2: dict) -> bool:
    """Checks the equivalence between 2 dictionaries, that can contain torch Tensors as value. The dictionary can be
    nested with other dictionaries or lists, they will be checked recursively.

    :param dict1: Dictionary to compare.
    :param dict2: Dictionary to compare.
    :return: True, if dict1 and dict2 are equal, false otherwise.
    """
    if len(dict1) != len(dict2):
        return False

    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        key_equal = k1 == k2
        value_equal = tensor_container_element_eq(v1, v2)

        if (not key_equal) or (not value_equal):
            return False

    return True


def tensor_list_eq(list1: list, list2: list) -> bool:
    """Checks the equivalence between 2 lists, that can contain torch Tensors as value. The list can be
    nested with other dictionaries and lists, they will be checked recursively. The dictionaries can have torch Tensors as value, not as key!

    :param list1: List to compare.
    :param list2: List to compare.
    :return: True, if list1 and list2 are equal, false otherwise.
    """
    if len(list1) != len(list2):
        return False

    for v1, v2 in zip(list1, list2):
        value_equal = tensor_container_element_eq(v1, v2)
        if not value_equal:
            return False

    return True


def tensor_container_element_eq(v1: Union[dict, list, th.Tensor, Any], v2: Union[dict, list, th.Tensor, Any]) -> bool:
    """ Checks equivalence between the two values and returns a single bool value, if the input is torch Tensor. If
    the input is a dictionary or list, it is recursively checked. The key of a (nested) dictionary can't be Tensor.

    :param v1: Value to compare. Can be dictionary, list, tensor or some other type that has the equal operator (==)
    return a single bool value.
    :param v2: Value to compare. Can be dictionary, list, tensor or some other type that
    has the equal operator (==) return a single bool value.
    :return: True, if v1 and v2 are equal, false otherwise.
    """
    if isinstance(v1, dict):
        return tensor_dict_eq(v1, v2)
    if isinstance(v1, list):
        return tensor_list_eq(v1, v2)
    elif isinstance(v1, th.Tensor):
        return th.all(v1 == v2)
    else:
        return v1 == v2
