

def disable_hf_loggers(is_local_main_process=True):
    if is_local_main_process:
        try:
            import transformers

            transformers.utils.logging.set_verbosity_warning()
        except:
            pass
        try:
            import diffusers

            diffusers.utils.logging.set_verbosity_warning()
        except:
            pass
    else:
        try:
            import transformers

            transformers.utils.logging.set_verbosity_error()
        except:
            pass
        try:
            import diffusers

            diffusers.utils.logging.set_verbosity_error()
        except:
            pass
