from invoke import task, call

@task
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building")

@task(pre=[build])
def run(c, k=0, n=0, Random=True):
    c.run("python3.8.5 main.py {} {} {}".format(k, n, Random))



