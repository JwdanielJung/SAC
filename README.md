### SAC (Search-Augmented unsafe prompts Classification) frameworks for LLMs

(1) Vector storing of unsafe prompts
(2) Threshold optimization 
(3) Similarity Search based threshold filtering
    - confi_unsafe: confident unsafe in filtering phase
    - confi_safe: confident safe in filtering phase
    - unconfident: can't determine
    - losses: incorrect filtering
(4) Classification for remain ones by using previous classification