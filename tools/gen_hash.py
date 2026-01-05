import bcrypt

pwd = input("Password: ").encode("utf-8")
h = bcrypt.hashpw(pwd, bcrypt.gensalt()).decode("utf-8")
print("\nBCRYPT HASH:\n", h)