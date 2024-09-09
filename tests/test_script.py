from src.agents.stepwise_agent import StepwiseCalculatorAgent


agent = StepwiseCalculatorAgent()

# ans = agent.run('10 + 5 * 3')

# ans = agent.run('10 + 5 * 3 - 8 / 2')

# ans = agent.run('10 + 5 * 3 - 8 / 2 * 3 + 1')

ans = agent.run('10.3 + 5.44 * 3.1 - 8.776 / 2.2 * 3.44 + 1.23')

print(ans)
