import numpy as np
from PIL import Image

class rgbAdjustment():
    '''
    Values and functions except the color Adjustment are adapted from
    the js code https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/l1c_optimized/
    from the Sentinel Hub Custom Script Repository transformed to Python.
    '''
    def __init__(self):
        # Constants used for image adjustments
        self.max_reflectance = 3.0
        self.midR = 0.13
        self.saturation = 1.2
        self.gamma = 1.8
        self.rayleigh = {'r': 0.013, 'g': 0.024, 'b': 0.041}
        self.gOff = 0.01
        self.gOffPow = self.gOff ** self.gamma
        self.gOffRange = (1 + self.gOff) ** self.gamma - self.gOffPow
        self.curve_points_x_y = np.array([[0.41121495327102803, 0], [0.64252336448598135, 0.62890625], [1, 1]])

    def adjust_reflectance(self, reflectance, midR, max_reflectance):
        reflectance = np.clip(reflectance / max_reflectance, 0, 1)
        return reflectance * (reflectance * (midR / max_reflectance + 1 - midR) - 1) / (reflectance * (2 * midR / max_reflectance - 1) - midR / max_reflectance)

    def adjust_gamma(self, reflectance):
        return ((reflectance + self.gOff) ** self.gamma - self.gOffPow) / self.gOffRange

    def saturation_enhancement(self, r, g, b):
        avg_saturation = (r + g + b) / 3.0 * (1 - self.saturation)
        r_sat = np.clip(avg_saturation + r * self.saturation, 0, 1)
        g_sat = np.clip(avg_saturation + g * self.saturation, 0, 1)
        b_sat = np.clip(avg_saturation + b * self.saturation, 0, 1)
        return r_sat, g_sat, b_sat

    def to_sRGB(self, channel):
        # Ensure channel is a numpy array
        channel = np.asarray(channel)

        # Vectorized operations for the whole array
        return np.where(channel <= 0.0031308,
                        12.92 * channel,
                        1.055 * np.power(channel, 1 / 2.4) - 0.055)
    
    def get_curve(self):
        '''
        Get value curve for image by three points (x, y) 0 - 1
        '''
        # 0 = a + b * 0.41121495327102803 + c * 0.41121495327102803^2
        # 0.62890625 = a + b * 0.64252336448598135 + c * 0.64252336448598135^2
        # 1 = a + b + c

        return np.linalg.solve(np.transpose([self.curve_points_x_y[:,0]**2, self.curve_points_x_y[:,0], [1, 1, 1]]), self.curve_points_x_y[:,1]) # c, b, a


    def adjust_colors(self, channel, curve_values):
        #curve values = [c, b, a] (c*x^2 + b*x + a)
        curve = np.poly1d(curve_values)
        return curve(channel)

    def evaluate_pixel(self, b04, b03, b02):
        r_lin = self.adjust_gamma(self.adjust_reflectance(b04, self.midR, self.max_reflectance))
        g_lin = self.adjust_gamma(self.adjust_reflectance(b03, self.midR, self.max_reflectance))
        b_lin = self.adjust_gamma(self.adjust_reflectance(b02, self.midR, self.max_reflectance))
        r_sat, g_sat, b_sat = self.saturation_enhancement(r_lin, g_lin, b_lin)
        image = []
        curve_values = self.get_curve()
        for channel in [r_sat, g_sat, b_sat]:
            s_rgb = self.to_sRGB(channel)
            image.append(self.adjust_colors(s_rgb, curve_values))
        rgb = np.dstack(image)
        return Image.fromarray((rgb*255).astype(np.uint8))
