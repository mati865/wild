// The x86 CET options are ignored when linking for AArch64: like bfd, Wild warns about each one and
// doesn't emit an x86 IBT property.
//#Arch:aarch64
//#Mode:static
//#RunEnabled:false
//#ReferenceLinkers:bfd
//#LinkArgs:-z ibt -z ibtplt -z force-ibt -z cet-report=warning
//#ExpectWarning:-z ibt ignored
//#ExpectWarning:-z ibtplt ignored
//#ExpectWarning:-z force-ibt ignored
//#ExpectWarning:-z cet-report.* ignored
//#NoSection:.note.gnu.property

.globl _start
_start:
  ret
