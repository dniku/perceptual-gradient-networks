from .model import AdaptiveInstanceNorm


class DecoderFromLatents:
    def __init__(self, generator, step=8):
        self.generator = generator
        self.step = step

    def __call__(self, latent_params):
        latent_w = latent_params['latent_w:0']
        for level_i, block in enumerate(self.generator.generator.progression):
            adain1_style = latent_params.get('latent_w_prime:{}_1'.format(level_i), None)
            if adain1_style is not None:
                block.adain1.fixed_style = [adain1_style]

            adain2_style = latent_params.get('latent_w_prime:{}_2'.format(level_i), None)
            if adain2_style is not None:
                block.adain2.fixed_style = [adain2_style]

        noise = [latent_params[f'noise:{i}'] for i in range(self.step + 1)]

        result = self.generator([latent_w], latent_type='w', step=self.step, noise=noise)

        for module in self.generator.modules():
            if isinstance(module, AdaptiveInstanceNorm):
                module.fixed_style = None

        return result
