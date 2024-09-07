from calculator_agent import CalculatorAgent


agent = CalculatorAgent()

ans = agent.run('10 + 5 * 3')

# ans = agent.run('10 + 5 * 3 - 8 / 2 * 3 + 1')

print(ans)
