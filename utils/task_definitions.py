import numpy as np

constraints = dict(
                    random_cloth = lambda min, mid, max, dist, origin_x,origin_y: [
                                        dict(origin=f"S{origin_x}_{origin_y}", 
                                             target=f"S{x}_{y}", 
                                             distance=np.sqrt(dist[x*y][0]**2 + dist[x*y][1]**2 + dist[x*y][2]**2))   # dist[x*y]: (x,y,z) 형식, 숫자 하나로 나오게 하려면 sqrt()
                                             for x in range(max + 1) for y in range(max + 1)
                                        ],

                    diagonal = lambda min, mid, max, dist: [
                                   dict(origin=f"S{max}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(1, 1, 0)),
                                   dict(origin=f"S{min}_{min}", target=f"S{min}_{min}",distance=dist),
                                   dict(origin=f"S{mid}_{mid}", target=f"S{mid}_{mid}", distance=dist), ],


                    sideways = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{min}", target=f"S{max}_{min}", distance=dist),
                                        dict(origin=f"S{min}_{min}", target=f"S{min}_{min}", distance=dist)],

                    sideways_two_corners = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_one_corner = lambda min, mid, max, dist: [
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0))],

                    sideways_two_corners_mid = lambda min, mid, max, dist: [
                                        dict(origin=f"S{max}_{max}", target=f"S{max}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{max}", target=f"S{min}_{min}", distance=dist, noise_directions=(0, 1, 0)),
                                        dict(origin=f"S{min}_{mid}", target=f"S{min}_{mid}", distance=dist),
                                        dict(origin=f"S{max}_{mid}", target=f"S{max}_{mid}", distance=dist)],
                   )
