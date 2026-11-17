load pose_1.pdb, complex
hide everything
show cartoon, complex
color gray80, chain A
color marine, chain H
color cyan, chain L
select interface, byres ((chain H+L within 4 of chain A) or (chain A within 4 of chain H+L))
show sticks, interface
set cartoon_transparency, 0.1
bg_color white
zoom complex
