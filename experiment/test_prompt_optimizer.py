
from prompt_optimizer.poptim import EntropyOptim

from prompt_optimizer.metric import TokenMetric

from prompt_optimizer.poptim import LemmatizerOptim

prompt = """Hi, I am Vikas Lakka and i am from India. Can you please name any 10 varieties of spices available all over the world?"""
p_optimizer = EntropyOptim(verbose=False, p=0.5, metrics= [TokenMetric()])
optimized_prompt = p_optimizer(prompt)
print(optimized_prompt)

p_optimizer = LemmatizerOptim(verbose=True, metrics=[TokenMetric()])
optimized_prompt = p_optimizer(prompt)
assert len(optimized_prompt) > 0, "Failed!"