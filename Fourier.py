from manim import *
class FourierScene(Scene):
    conf= {
        'slow_factor': .5,
        'wait_before_start': None,
        'texto': r'\rm R',
        'n_vectors': 10,
        'center_point': ORIGIN,
        'vector_config': {
            'buff': 0,
            'max_tip_length_to_length_ratio': 0.25,
            'tip_length': 0.15,
            'max_stroke_width_to_length_ratio': 10,
            'stroke_width': 1.7,
        }
    }
    def setup(self):
        self.slow_factor_tracker= ValueTracker()
        self.vector_clock=ValueTracker()
    def construct(self):
        self.add_vector_circles_path()
        if self.conf['wait_before_start'] is not None:
            self.wait(self.conf['wait_before_start'])
        else:
            self.wait(1)
    def add_vector_clock(self):
        self.vector_clock.add_updater(
            lambda m, dt: m.increment_value(self.conf['slow_factor'])
        )
    def add_vector_circles_path(self):
        path= self.get_path()
        coeffs= self.get_coefficients_of_path(path)
        vectors= self.get_rotating_vectors()
        self.add(
            path,
            vectors
        )
    def get_freqs(self):
        n= self.conf['n_vectors']
        all_freqs= list(range(n//2, -n//2, -1))
        all_freqs.sort(key=abs)
        return all_freqs
    def get_coefficients(self):
        return [
            complex(0) for _ in range(self.conf['n_vectors'])
        ]
    def get_path(self):
        text_mob= MathTex(self.conf['texto'])
        text_mob.set_height(6)
        path= text_mob.family_members_with_points()[0]
        return path
    def get_coefficients_of_path(self, path, n_samples= 10000, freqs= None):
        if freqs is None:
            freqs= self.get_freqs()
        dt= 1/n_samples
        ts= np.arange(0,1, dt)
        samples= np.array([path.point_from_proportion(t) for t in ts ])
        samples-= self.conf['center_point']
        complex_samples= samples[:,0]+ 1j *samples[:,1]
        return [
            np.array([
                np.exp(-TAU*1j*freq*t) * cs for t, cs in zip(ts, complex_samples)
            ]).sum()*dt for freq in freqs
        ]
    def get_rotating_vectors(self, freqs=None, coefficients=None):
        vectors= VGroup()
        self.center_tracker= VectorizedPoint(self.conf['center_point'])
        if freqs is None:
            freqs= self.get_freqs()
        if coefficients is None:
            coefficients= self.get_coefficients()
        last_vector= None
        for freq, coefficient in zip(freqs, coefficients):
            if last_vector:
                center_func= last_vector.get_end
            else:
                center_func= self.center_tracker.get_location
            vector= self.get_rotating_vector(
                freq, coefficient, center_func
            )
            vectors.add(vector)
            last_vector= vector
        return vectors
    def get_rotating_vector(self, freq, coefficient, center_func):
        vector = Vector(RIGHT, **self.conf['vector_config'])
        vector.scale(abs(coefficient))
        if abs(coefficient) == 0:
            phase = 0
        else:
            phase = np.log(coefficient).imag
        vector.rotate(phase, about_point=ORIGIN)
        vector.freq = freq
        vector.coefficient = coefficient
        vector.center_func = center_func
        vector.add_updater(self.update_rotating_vector)
        return vector
    def update_rotating_vector(self, vector, dt):
        time= self.vector_clock.get_value()
        coef = vector.coefficient
        freq = vector.freq
        phase= np.log(coef).imag
        vector.set_length(np.max(abs(coef)))
        vector.set_angle(
            phase +time*TAU*freq,
        )
        vector.shift(vector.center_func() - vector.get_start())
        return vector