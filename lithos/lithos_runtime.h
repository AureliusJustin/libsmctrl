/*
 * LithOS runtime wrapper interface.
 *
 * Exposes lifecycle hooks called from the libsmctrl wrapper constructor and
 * destructor when LIBSMCTRL_WRAPPER is enabled.
 */
#pragma once

#ifdef LIBSMCTRL_WRAPPER
void lithos_wrapper_init(void);
void lithos_wrapper_shutdown(void);
#endif
