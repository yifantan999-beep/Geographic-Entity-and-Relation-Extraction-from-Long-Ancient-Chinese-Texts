mods=["latex2mathml","mathml2omml","lxml","sympy","docx"]
for name in mods:
    try:
        m=__import__(name); print(name,"yes",getattr(m,"__version__",""))
    except Exception as e: print(name,"no",type(e).__name__,str(e)[:120])
