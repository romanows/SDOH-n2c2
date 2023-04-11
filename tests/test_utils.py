# pylint: disable=missing-docstring

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from n2c2_ss.utils import strip_lines, normalize_for_t5_tokenizer, postprocess_t5_decoded
import unittest


class Test(unittest.TestCase):


    def test_strip_lines(self):
        self.assertEqual("", strip_lines(""))
        self.assertEqual("", strip_lines(" "))
        self.assertEqual("", strip_lines("\n"))
        self.assertEqual("", strip_lines("\n \n"))

        self.assertEqual("foo bar", strip_lines("foo bar"))
        self.assertEqual("foo\nbar", strip_lines("foo\nbar"))
        self.assertEqual("foo\n\nbar", strip_lines("foo\n\nbar"))
        self.assertEqual("foo\n\nbar", strip_lines("foo\n   \nbar"))
        self.assertEqual("foo\n\nbar", strip_lines("    foo   \n   \n   bar   "))


    def test_normalize_for_t5_tokenizer(self):
        self.assertEqual("", normalize_for_t5_tokenizer(""))
        self.assertEqual("", normalize_for_t5_tokenizer(" "))
        self.assertEqual("", normalize_for_t5_tokenizer(" \n \n "))

        self.assertEqual("foo bar", normalize_for_t5_tokenizer("foo bar"))
        self.assertEqual("foo bar", normalize_for_t5_tokenizer("   foo  bar "))

        self.assertEqual("foo €n bar", normalize_for_t5_tokenizer("foo\nbar"))

        # Note how the spaces around the newline are eliminated since they appear at the
        # beginning or the ending of a newline.
        self.assertEqual("foo €n bar", normalize_for_t5_tokenizer("foo \nbar"))
        self.assertEqual("foo €n bar", normalize_for_t5_tokenizer("foo\n bar"))
        self.assertEqual("foo €n bar", normalize_for_t5_tokenizer("foo \n bar"))

        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo\n\nbar"))
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo \n\nbar"))
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo \n \nbar"))
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo \n \n bar"))
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo\n \n bar"))
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo\n\n bar"))

        # Collapse multiple blank lines to just one blank line
        self.assertEqual("foo €n €n bar", normalize_for_t5_tokenizer("foo\n\n\nbar"))  # two blank lines -> one blank line

        # The _NOT_A_SPACE comes into play for non-newline special characters
        self.assertEqual("foo ® > ® bar", normalize_for_t5_tokenizer("foo>bar"))
        self.assertEqual("foo > ® bar", normalize_for_t5_tokenizer("foo >bar"))
        self.assertEqual("foo ® > bar", normalize_for_t5_tokenizer("foo> bar"))
        self.assertEqual("foo > bar", normalize_for_t5_tokenizer("foo > bar"))

        self.assertEqual("foo ® > ® > ® bar", normalize_for_t5_tokenizer("foo>>bar"))
        self.assertEqual("foo > ® > ® bar", normalize_for_t5_tokenizer("foo >>bar"))
        self.assertEqual("foo ® > ® > bar", normalize_for_t5_tokenizer("foo>> bar"))
        self.assertEqual("foo > ® > bar", normalize_for_t5_tokenizer("foo >> bar"))

        self.assertEqual("foo ® > > ® bar", normalize_for_t5_tokenizer("foo> >bar"))
        self.assertEqual("foo > > ® bar", normalize_for_t5_tokenizer("foo > >bar"))
        self.assertEqual("foo ® > > bar", normalize_for_t5_tokenizer("foo> > bar"))
        self.assertEqual("foo > > bar", normalize_for_t5_tokenizer("foo > > bar"))

        # Curly-braces are a bit of a special case since they end up being two characters
        # wide.  Tests are the same as above but the sequential special characters end up
        # being different characters, "<" and then ">".
        self.assertEqual("foo ® €C ® bar", normalize_for_t5_tokenizer("foo{bar"))
        self.assertEqual("foo €C ® bar", normalize_for_t5_tokenizer("foo {bar"))
        self.assertEqual("foo ® €C bar", normalize_for_t5_tokenizer("foo{ bar"))
        self.assertEqual("foo €C bar", normalize_for_t5_tokenizer("foo { bar"))

        self.assertEqual("foo ® €C ® €c ® bar", normalize_for_t5_tokenizer("foo{}bar"))
        self.assertEqual("foo €C ® €c ® bar", normalize_for_t5_tokenizer("foo {}bar"))
        self.assertEqual("foo ® €C ® €c bar", normalize_for_t5_tokenizer("foo{} bar"))
        self.assertEqual("foo €C ® €c bar", normalize_for_t5_tokenizer("foo {} bar"))

        self.assertEqual("foo ® €C €c ® bar", normalize_for_t5_tokenizer("foo{ }bar"))
        self.assertEqual("foo €C €c ® bar", normalize_for_t5_tokenizer("foo { }bar"))
        self.assertEqual("foo ® €C €c bar", normalize_for_t5_tokenizer("foo{ } bar"))
        self.assertEqual("foo €C €c bar", normalize_for_t5_tokenizer("foo { } bar"))


    def test_postprocess_t5_decoded(self):
        self.assertEqual("", postprocess_t5_decoded(""))

        self.assertEqual("foo bar", postprocess_t5_decoded("foo bar"))

        self.assertEqual("foo\nbar", postprocess_t5_decoded("foo €n bar"))
        self.assertEqual("foo\n\nbar", postprocess_t5_decoded("foo €n €n bar"))

        # normalize_for_t5_tokenizer doesn't output runs of more than two newlines but
        # it's fair game for the model output.  We still output a whitespaced-normalized
        # version of the input.
        self.assertEqual("foo\n\nbar", postprocess_t5_decoded("foo €n €n €n bar"))

        # Similar to above, we normalize/strip the leading and trailing whitespace.
        self.assertEqual("foo\nbar", postprocess_t5_decoded("€n foo €n bar €n"))

        # normalize_for_t5_tokenizer doesn't output the _NOT_A_SPACE character by itself
        # but who knows what we'll see in the model output.
        self.assertEqual("foobar", postprocess_t5_decoded("foo ® bar"))

        # The _NOT_A_SPACE comes into play for non-newline special characters
        self.assertEqual("foo>bar", postprocess_t5_decoded("foo ® > ® bar"))
        self.assertEqual("foo >bar", postprocess_t5_decoded("foo > ® bar"))
        self.assertEqual("foo> bar", postprocess_t5_decoded("foo ® > bar"))
        self.assertEqual("foo > bar", postprocess_t5_decoded("foo > bar"))

        self.assertEqual("foo>>bar", postprocess_t5_decoded("foo ® > ® > ® bar"))
        self.assertEqual("foo >>bar", postprocess_t5_decoded("foo > ® > ® bar"))
        self.assertEqual("foo>> bar", postprocess_t5_decoded("foo ® > ® > bar"))
        self.assertEqual("foo >> bar", postprocess_t5_decoded("foo > ® > bar"))

        self.assertEqual("foo> >bar", postprocess_t5_decoded("foo ® > > ® bar"))
        self.assertEqual("foo > >bar", postprocess_t5_decoded("foo > > ® bar"))
        self.assertEqual("foo> > bar", postprocess_t5_decoded("foo ® > > bar"))
        self.assertEqual("foo > > bar", postprocess_t5_decoded("foo > > bar"))

        # Curly-braces are a bit of a special case since they end up being two characters
        # wide.  Tests are the same as above but the sequential special characters end up
        # being different characters, "<" and then ">".
        self.assertEqual("foo{bar", postprocess_t5_decoded("foo ® €C ® bar"))
        self.assertEqual("foo {bar", postprocess_t5_decoded("foo €C ® bar"))
        self.assertEqual("foo{ bar", postprocess_t5_decoded("foo ® €C bar"))
        self.assertEqual("foo { bar", postprocess_t5_decoded("foo €C bar"))

        self.assertEqual("foo{}bar", postprocess_t5_decoded("foo ® €C ® €c ® bar"))
        self.assertEqual("foo {}bar", postprocess_t5_decoded("foo €C ® €c ® bar"))
        self.assertEqual("foo{} bar", postprocess_t5_decoded("foo ® €C ® €c bar"))
        self.assertEqual("foo {} bar", postprocess_t5_decoded("foo €C ® €c bar"))

        self.assertEqual("foo{ }bar", postprocess_t5_decoded("foo ® €C €c ® bar"))
        self.assertEqual("foo { }bar", postprocess_t5_decoded("foo €C €c ® bar"))
        self.assertEqual("foo{ } bar", postprocess_t5_decoded("foo ® €C €c bar"))
        self.assertEqual("foo { } bar", postprocess_t5_decoded("foo €C €c bar"))


if __name__ == "__main__":
    unittest.main()
