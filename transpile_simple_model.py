import joblib as jb

model = jb.load('regression_trained.joblib') # load linear regression

coefs = model.coef_ # get model coefs
n_thetas = len(coefs)

c_code_prediction = """
float fact(float i){
    if (i <= 1)
        return 1;
    else
        return i * fact(i - 1);
};

float power(float x, float n)
{
    if (n == 0)
        return 1;
    else
        return x * power(x, n - 1);
}


float exp_approx(float x, int n_term)
{
    float res = 0;
    for (int i = 0; i <= n_term; i++)
    {
        res += power(x, i) / fact(i);
    }
    return res;
}

float sigmoid(float x)
{
    return 1 / (1 + exp_approx(-x, 10));
}

float logistic_regression_prediction(float* features, float* thetas, int n_parameter) {
    float res = 0;

    for (int i = 0; i < n_parameter; ++i) {
        res += features[i] * thetas[i + 1];
    }

    res += thetas[0];
    return sigmoid(res);
}

float linear_regression_prediction(float* features, float* thetas, int n_thetas)
{
    float res = 0;

    for (int i = 0; i < n_thetas; ++i) {
        res += features[i] * thetas[i + 1];
    }
    res += thetas[0];

    return res;
}
"""


c_coefs_array = "{"
for i in range(n_thetas - 1):
    c_coefs_array += str(coefs[i]) + ','
c_coefs_array += str(coefs[n_thetas - 1]) + '}'

c_coefs = f"float thetas[{n_thetas}] = {c_coefs_array};"

c_main = f"""
#include <stdio.h>

{c_code_prediction}

int main(int argc, char *argv[])
{{
    {c_coefs}
    int n_thetas = {n_thetas};
    float features[2] = {{-0.004164936524136716,0.0017850734344602727}};
    float pred = logistic_regression_prediction(features, thetas, n_thetas);

    printf("%f", pred);
}}
"""

# save c code
c_file = open('pred.c', 'w')
c_file.write(c_main)
c_file.close()

print('run <gcc pred.c> to compile c file')

