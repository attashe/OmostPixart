from comfy import model_management


class ModelUnloader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "latent": ("LATENT",),                
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "unload_model"

    CATEGORY = "LJRE/utils"

    def unload_model(self, latent):
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
        return (latent,)
    

NODE_CLASS_MAPPINGS = {
    "ModelUnloader": ModelUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelUnloader": "Unload Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
