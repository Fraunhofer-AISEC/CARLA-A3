from cgitb import text
import carla
import numpy as np
from matplotlib import pyplot


class Injector():
    def __init__(self) -> None:
        pass

    @classmethod
    def inject(
        cls, 
        client:carla.Client, 
        texture_img : np.array or str = None
    ) -> None:
        
        world = client.get_world()

        if texture_img is None:
            texture_img = np.zeros((128, 128, 4)).astype(np.uint8)
        
        elif isinstance(texture_img, np.ndarray):
            texture_img = [np.fliplr(texture_img[:,:,i]) for i in range(4)]
            texture_img = np.stack(texture_img, 2)
            texture_img = (texture_img * 255).astype(np.uint8)

        elif isinstance(texture_img, str):
            texture_img = pyplot.imread(texture_img)
            texture_img = [np.fliplr(texture_img[:,:,i]) for i in range(4)]
            texture_img = np.stack(texture_img, 2)
            texture_img = (texture_img * 255).astype(np.uint8)
            # If the image needs to be rotated
            texture_img = np.rot90(texture_img,1,(0,1)) 

        h, w, _ = texture_img.shape
        texture = carla.TextureColor(h, w)
        for i in range(h):
            for j in range(w):
                texture.set(
                    i, j, 
                    carla.Color(
                        r=texture_img[i,j,0].item(),
                        g=texture_img[i,j,1].item(),
                        b=texture_img[i,j,2].item(),
                        a=texture_img[i,j,3].item(),
                    )
                )
        # Change ID Name value if it is different inside Carla UI
        world.apply_color_texture_to_object( 
            "BP_AdvStopSign_2",
            carla.MaterialParameter.Diffuse,
            texture
        )
