KPL/MK

    Define the paths to the kernel directory. Use the PATH_SYMBOLS
    as aliases to the paths.

    The names and contents of the kernels referenced by this
    meta-kernel are as follows:

    File Name        Description
    ---------------  ------------------------------
    naif0008.tls     Generic LSK.
    de405s.bsp       Planet Ephemeris SPK.
    pck00008.tpc     Generic PCK.


\begindata

    PATH_VALUES     = ( 'lessons/other_stuff/kernels/lsk',
                        'lessons/other_stuff/kernels/spk',
                        'lessons/other_stuff/kernels/pck' )

    PATH_SYMBOLS    = ( 'LSK', 'SPK', 'PCK' )

    KERNELS_TO_LOAD = ( '$LSK/naif0008.tls',
                        '$SPK/de405s.bsp',
                        '$PCK/pck00008.tpc' )

\begintext