for mod in ["fitz","pdf2image","PIL","docx"]:
    try:
        m=__import__(mod); print(mod,"ok",getattr(m,"__version__",""))
    except Exception as e: print(mod,"no",e)
