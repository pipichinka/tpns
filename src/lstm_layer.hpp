#include "layer.hpp"


class lstm_layer: public layer_interface{
    int input_size; 
    int hidden_size;
    int output_size;
    nc::NdArray<double> Wi;
    nc::NdArray<double> Ui;
    nc::NdArray<double> bi;

    nc::NdArray<double> Wf;
    nc::NdArray<double> Uf;
    nc::NdArray<double> bf;

    nc::NdArray<double> Wo;
    nc::NdArray<double> Uo;
    nc::NdArray<double> bo;

    nc::NdArray<double> Wc;
    nc::NdArray<double> Uc;
    nc::NdArray<double> bc;

    nc::NdArray<double> Wy;
    nc::NdArray<double> by;


    nc::NdArray<double> hs;
    nc::NdArray<double> cs;
    nc::NdArray<double> ins;
    nc::NdArray<double> os;
    nc::NdArray<double> fs;
    nc::NdArray<double> _cs;
    nc::NdArray<double> c_tanh;

    nc::NdArray<double> xs;
public:

    lstm_layer(int input_size, int hidden_size, int output_size, double ws):
    input_size(input_size),
    hidden_size(hidden_size),
    output_size(output_size)
    {
        Wi = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Ui = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bi = nc::zeros<double>(1, hidden_size);

        Wf = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Uf = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bf = nc::zeros<double>(1, hidden_size);

        Wo = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Uo = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bo = nc::zeros<double>(1, hidden_size);

        Wc = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Uc = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bc = nc::zeros<double>(1, hidden_size);

        Wy = nc::random::uniform<double>(nc::Shape(hidden_size, output_size), -ws, ws);
        by = nc::zeros<double>(1, output_size);
    }
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        xs = x;
        int rows = x.numRows();
        nc::NdArray<double> ht = nc::zeros<double>(1, hidden_size);
        nc::NdArray<double> ct = nc::zeros<double>(1, hidden_size);
        
        hs = nc::NdArray<double>(rows, hidden_size);
        cs = nc::NdArray<double>(rows, hidden_size);
        ins = nc::NdArray<double>(rows, hidden_size);
        os = nc::NdArray<double>(rows, hidden_size);
        fs = nc::NdArray<double>(rows, hidden_size);
        _cs = nc::NdArray<double>(rows, hidden_size);
        c_tanh = nc::NdArray<double>(rows, hidden_size);
        for (int t = 0; t < rows; t++){
            nc::NdArray<double> xt = x.row(t);
            nc::NdArray<double> it = activation_sigmoid::forward( nc::dot(xt, Wi) + nc::dot(ht, Ui) + bi);
            nc::NdArray<double> ft = activation_sigmoid::forward( nc::dot(xt, Wf) + nc::dot(ht, Uf) + bf);
            nc::NdArray<double> ot = activation_sigmoid::forward( nc::dot(xt, Wo) + nc::dot(ht, Uo) + bo);
            nc::NdArray<double> _c = activation_tanh::forward( nc::dot(xt, Wc) + nc::dot(ht, Uc) + bc);

            ins.put(t, nc::Slice(0, hidden_size), it);
            os.put(t, nc::Slice(0, hidden_size), ot);
            fs.put(t, nc::Slice(0, hidden_size), ft);
            _cs.put(t, nc::Slice(0, hidden_size), _c);
            ct = ft * ct + it * _c;
            cs.put(t, nc::Slice(0, hidden_size), ct);
            nc::NdArray<double> c_tanht = activation_tanh::forward(ct);
            c_tanh.put(t, nc::Slice(0, hidden_size), c_tanht);
            ht = ot * c_tanht;
            hs.put(t, nc::Slice(0, hidden_size), ht);
        }

        return nc::dot(hs, Wy) + by; 
    }

    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){
        int rows = error.numRows();
        nc::NdArray<double> dWi = nc::zeros_like<double>(Wi);
        nc::NdArray<double> dUi = nc::zeros_like<double>(Ui);
        nc::NdArray<double> dbi = nc::zeros_like<double>(bi);

        nc::NdArray<double> dWf = nc::zeros_like<double>(Wf);
        nc::NdArray<double> dUf = nc::zeros_like<double>(Uf);
        nc::NdArray<double> dbf = nc::zeros_like<double>(bf);

        nc::NdArray<double> dWo = nc::zeros_like<double>(Wo);
        nc::NdArray<double> dUo = nc::zeros_like<double>(Uo);
        nc::NdArray<double> dbo = nc::zeros_like<double>(bo);

        nc::NdArray<double> dWc = nc::zeros_like<double>(Wc);
        nc::NdArray<double> dUc = nc::zeros_like<double>(Uc);
        nc::NdArray<double> dbc = nc::zeros_like<double>(bc);

        nc::NdArray<double> dWy = nc::zeros_like<double>(Wy);
        nc::NdArray<double> dby = nc::zeros_like<double>(by);

        nc::NdArray<double> dh_next = nc::zeros<double>(1, hidden_size);
        nc::NdArray<double> dc_next = nc::zeros<double>(1, hidden_size);

        nc::NdArray<double> dx = nc::zeros<double>(rows, input_size);

        for (int t = rows - 1; t >=0; t--){
            nc::NdArray<double> dy = error.row(t);
            dWy += nc::dot(hs.row(t).transpose(), dy);
            dby += dy;

            nc::NdArray<double> dh = nc::dot(dy, Wy.transpose()) + dh_next;
            nc::NdArray<double> dc = os.row(t) * dh * activation_tanh::derivative(cs.row(t)) + dc_next;

            nc::NdArray<double> dot = activation_sigmoid::derivative(os.row(t) * c_tanh.row(t) * dh);

            nc::NdArray<double> dft;
            if (t > 0){
                dft = cs.row(t - 1) * dc * activation_sigmoid::derivative(fs.row(t));
            }
            else {
                dft = nc::zeros<double>(1, hidden_size);
            }

            nc::NdArray<double> dit = _cs.row(t) * dc * activation_sigmoid::derivative(ins.row(t)); 
            nc::NdArray<double> dct = ins.row(t) * dc * activation_tanh::derivative(_cs.row(t));
            nc::NdArray<double> xt = xs.row(t).transpose();
            dWi += nc::dot(xt, dit);
            dbi += dit;

            dWf += nc::dot(xt, dft);
            dbf += dft;

            dWo += nc::dot(xt, dot);
            dbo += dot;

            dWc += nc::dot(xt, dct);
            dbc += dct;

            if (t > 0){
                nc::NdArray<double> hst = hs.row(t - 1).transpose();
                dUi += nc::dot(hst, dit);
                dUf += nc::dot(hst, dft);
                dUo += nc::dot(hst, dot);
                dUc += nc::dot(hst, dct);
            }

            dh_next = nc::dot(dit, Ui.transpose()) + nc::dot(dft, Uf.transpose()) + nc::dot(dot, Uo.transpose()) + nc::dot(dct, Uc.transpose());
            dc_next = fs.row(t) * dc;
            dx.put(t, nc::Slice(0, input_size), nc::dot(dit, Wi.transpose()) + nc::dot(dft, Wf.transpose()) + nc::dot(dot, Wo.transpose()) + nc::dot(dct, Wc.transpose()));
        }

        Wi -= learn_rate * dWi;
        Ui -= learn_rate * dUi;
        bi -= learn_rate * dbi;
        Wf -= learn_rate * dWf;
        Uf -= learn_rate * dUf;
        bf -= learn_rate * dbf;
        Wo -= learn_rate * dWo;
        Uo -= learn_rate * dUo;
        bo -= learn_rate * dbo;
        Wc -= learn_rate * dWc;
        Uc -= learn_rate * dUc;
        bc -= learn_rate * dbc;
        Wy -= learn_rate * dWy;
        by -= learn_rate * dby;
        return dx;
    }

};


class relu_layer:public layer_interface{
    nc::NdArray<double> sx;
public:
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        sx = x;
        nc::applyFunction(sx , std::function([](double v) -> double{
            return std::max(0.0, v);
        }));

        return sx;
    }

    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){
        nc::applyFunction(sx , std::function([](double v) -> double{
            return v >= 0.0 ? 1.0: 0.0;
        }));
        return error * sx ;
    }
};