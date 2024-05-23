#include "layer.hpp"

class gru_layer: public layer_interface{
    int input_size; 
    int hidden_size;
    int output_size;

    nc::NdArray<double> Wz;
    nc::NdArray<double> Uz;
    nc::NdArray<double> bz;

    nc::NdArray<double> Wr;
    nc::NdArray<double> Ur;
    nc::NdArray<double> br;

    nc::NdArray<double> Wh;
    nc::NdArray<double> Uh;
    nc::NdArray<double> bh;

    nc::NdArray<double> Wy;
    nc::NdArray<double> by;

    nc::NdArray<double> x;
    nc::NdArray<double> z;
    nc::NdArray<double> r;
    nc::NdArray<double> h;
    nc::NdArray<double> h_hat;

public:
    gru_layer(int input_size, int hidden_size, int output_size, double ws):
    input_size(input_size),
    hidden_size(hidden_size),
    output_size(output_size)
    {
        Wz = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Uz = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bz = nc::zeros<double>(1, hidden_size);

        Wr = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Ur = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        br = nc::zeros<double>(1, hidden_size);

        Wh = nc::random::uniform<double>(nc::Shape(input_size, hidden_size), -ws, ws);
        Uh = nc::random::uniform<double>(nc::Shape(hidden_size, hidden_size), -ws, ws);
        bh = nc::zeros<double>(1, hidden_size);

        Wy = nc::random::uniform<double>(nc::Shape(hidden_size, output_size), -ws, ws);
        by = nc::zeros<double>(1, output_size);

    }
    virtual nc::NdArray<double> forward(const nc::NdArray<double>& x){
        this->x = x;
        int rows = x.numRows();
        h = nc::zeros<double>(rows, hidden_size);
        z = nc::zeros<double>(rows, hidden_size);
        r = nc::zeros<double>(rows, hidden_size);
        h_hat = nc::zeros<double>(rows, hidden_size);

        nc::NdArray<double> prev_h = nc::zeros<double>(1, hidden_size);
        for (int t = 0; t < rows; t++){
            nc::NdArray<double> xt = x.row(t);
            z.put(t, nc::Slice(0, hidden_size), activation_sigmoid::forward(nc::dot(xt, Wz) + nc::dot(prev_h, Uz) + bz));
            r.put(t, nc::Slice(0, hidden_size), activation_sigmoid::forward(nc::dot(xt, Wr) + nc::dot(prev_h, Ur) + br));
            h_hat.put(t, nc::Slice(0, hidden_size), activation_tanh::forward(nc::dot(xt, Wh) + nc::dot(r.row(t) * prev_h,Uh) + bh));
            h.put(t, nc::Slice(0, hidden_size), z.row(t) * prev_h + (1.0 - z.row(t)) * h_hat.row(t));
        }

        return nc::dot(h, Wy) + by;
    }

    virtual nc::NdArray<double> backward(const nc::NdArray<double> error, double learn_rate){
        int rows = error.numRows();
        nc::NdArray<double> dWz = nc::zeros_like<double>(Wz);
        nc::NdArray<double> dUz = nc::zeros_like<double>(Uz);
        nc::NdArray<double> dbz = nc::zeros_like<double>(bz);

        nc::NdArray<double> dWr = nc::zeros_like<double>(Wr);
        nc::NdArray<double> dUr = nc::zeros_like<double>(Ur);
        nc::NdArray<double> dbr = nc::zeros_like<double>(br);

        nc::NdArray<double> dWh = nc::zeros_like<double>(Wh);
        nc::NdArray<double> dUh = nc::zeros_like<double>(Uh);
        nc::NdArray<double> dbh = nc::zeros_like<double>(bh);

        nc::NdArray<double> dWy = nc::zeros_like<double>(Wy);
        nc::NdArray<double> dby = nc::zeros_like<double>(by);

        nc::NdArray<double> dh_next = nc::zeros<double>(1, hidden_size);

        nc::NdArray<double> dx = nc::zeros<double>(rows, input_size);
        for (int t = rows - 1; t >= 0; t--){
            nc::NdArray<double> dy = error.row(t);
            dWy += nc::dot(h.row(t).transpose(), dy);
            dby += dy;
            nc::NdArray<double> dh = nc::dot(dy, Wy.transpose()) + dh_next;
            nc::NdArray<double> dh_hat = dh * (1.0 - z.row(t));
            nc::NdArray<double> dh_hat_l = dh_hat * activation_tanh::derivative(h_hat.row(t));

            dWh += nc::dot(x.row(t).transpose(), dh_hat_l); //1X24
            if (t > 0){
                dUh += nc::dot(r.row(t) * h.row(t - 1), dh_hat_l);
            }
            dbh += dh_hat_l;

            nc::NdArray<double> drhp = nc::dot(dh_hat_l, Uh.transpose());
            nc::NdArray<double> dr;
            if (t > 0){
                dr = drhp * h.row(t - 1);
            } else {
                dr = nc::zeros<double>(1, hidden_size);
            }
            nc::NdArray<double> dr_l = dr * activation_sigmoid::derivative(r.row(t));


            dWr +=  nc::dot(x.row(t).transpose(), dr_l);
            if (t > 0){
                dUr += nc::dot(h.row(t - 1).transpose(), dr_l);
            }
            dbr += dr_l;
            
            nc::NdArray<double> dz;
            if (t > 0){
                dz = dh * (h.row(t - 1) - h_hat.row(t));
            } else {
                dz = dh * (-h_hat.row(t));
            }
            nc::NdArray<double> dz_l = dz * activation_sigmoid::derivative(z.row(t));
            dWz += nc::dot(x.row(t).transpose(), dz_l);

            if (t > 0){
                dUz += nc::dot(h.row(t - 1), dz_l);
            }
            dbz += dz_l;

            nc::NdArray<double> dh_fz_inner = nc::dot(dz_l, Uz.transpose());
            nc::NdArray<double> dh_fz = dh * z.row(t);
            nc::NdArray<double> dh_fhh =  drhp * r.row(t);
            nc::NdArray<double> dh_fr = nc::dot(dr_l, Ur.transpose());
            dh_next = dh_fz_inner + dh_fz + dh_fhh + dh_fr;
            dx.put(t, nc::Slice(0, input_size), nc::dot(dz_l, Wz.transpose()) + nc::dot(dr_l, Wr.transpose()) + nc::dot(dh_hat_l, Wh.transpose()));
        }
        Wz -= learn_rate * dWz;
        Uz -= learn_rate * dUz;
        bz -= learn_rate * dbz;
        Wr -= learn_rate * dWr; 
        Ur -= learn_rate * dUr; 
        br -= learn_rate * dbr; 
        Wh -= learn_rate * dWh; 
        Uh -= learn_rate * dUh;
        bh -= learn_rate * dbh; 
        Wy -= learn_rate * dWy; 
        by -= learn_rate * dby; 


        return dx;
    }
};