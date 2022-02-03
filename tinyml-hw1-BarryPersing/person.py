class person:
    name = "Barry"
    age_yrs = 24
    height = 183
    
    def __init__(self, name, age, height):
        self.name = name  
        self.age_yrs = age
        self.height_cm = height

    def __repr__(self):
        rep = "{:} is {:} years old and {:} cm tall.".format(self.name, self.age_yrs, self.height)
        return rep


new_person = person(name='Joe',age=34,height=184)
print("{:} is {:} years old.".format(new_person.name, new_person.age_yrs))
print(repr(new_person))
barry = person(name='Barry', age=24, height=183)
print(repr(barry))
