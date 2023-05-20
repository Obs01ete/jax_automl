import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from constraints import Constraints


def transpose_list_dict(ld: List[Dict]) -> Optional[Dict[str, List]]:
    return {k: [d[k] for d in ld]
            for k in ld[0].keys()} if len(ld) > 0 else None


def visualize_results(lat_dicts: List[Dict],
                      seed_lat_dicts: List[Dict],
                      test_lat_dicts: List[Dict],
                      constraints: Constraints):

    opt_dict = transpose_list_dict(lat_dicts)
    seed_dict = transpose_list_dict(seed_lat_dicts)
    test_dict = transpose_list_dict(test_lat_dicts)

    plt.figure()
    plt.scatter(opt_dict['predicted_lat'], opt_dict['measured_lat'])
    plt.xlabel('pred_lats')
    plt.ylabel('measured_lats')
    plt.savefig('pred_vs_measured.png')

    plt.figure()
    x0 = constraints.parameters.min
    x1 = constraints.parameters.max
    y0 = constraints.latency_sec.min
    y1 = constraints.latency_sec.max
    plt.scatter(opt_dict['num_weights'], opt_dict['predicted_lat'],
                c='blue', label='optimized points')
    plt.scatter(seed_dict['num_weights'], seed_dict['predicted_lat'],
                c='gray', label='seed points')
    if test_dict is not None:
        plt.scatter(test_dict['num_weights'], test_dict['predicted_lat'],
                    c='red', label='known solution')
    plt.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0],
             'g--', label='target region')
    plt.xlabel('num_parameters')
    plt.ylabel('predicted_lats')
    plt.legend()
    plt.savefig('target_region.png')
