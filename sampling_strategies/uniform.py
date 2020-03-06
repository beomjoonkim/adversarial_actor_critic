from sampling_strategy import SamplingStrategy


class Uniform(SamplingStrategy):
    def __init__(self, environment, pick_pi, place_pi):
        SamplingStrategy.__init__(self, environment, pick_pi, place_pi)

    def sample_next_point(self, node):
        obj = node.obj
        region = node.region
        operator = node.operator
        if operator == 'two_arm_pick':
            action = self.pick_pi.predict(obj, region)
        elif operator == 'two_arm_place':
            action = self.place_pi.predict(obj, region)
        elif operator == 'one_arm_pick':
            action = self.one_arm_pick_pi.predict(obj, region)


        return action
