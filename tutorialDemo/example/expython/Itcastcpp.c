#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int fac(int n)
{
    if (n < 2)
        return 1;

    return n * fac(n - 1);
}

char *reverse(char *s)
{
    char t, *p = s, *q = (s + (strlen(s)-1));

    while (s && (p < q)) {
        t = *p;
        *p++ = *q;
        *q-- = t;
    }

    return s;
}

//int main(void)
int test(void)
{
    char s[1024];

    printf("4! == %d\n", fac(4));
    printf("8! == %d\n", fac(8));

    strcpy(s, "itcastcpp");
    printf("reversing 'itcastcpp', we get '%s'\n", reverse(s));

    return 0;
}
