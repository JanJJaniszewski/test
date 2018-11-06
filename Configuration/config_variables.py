foo = 'bar'

# Transforming variables to dictionary if use of feature dictionary preferred to use of original variables
variable_dict = {name: var for name, var in locals().items() if not name.startswith('__')}

if __name__ == '__main__':
    print(variable_dict)