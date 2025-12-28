module amul8x8 #(parameter DROP = 0)(
    input signed [7:0] x,
    input signed [7:0] y,
    output signed [15:0] exact,
    output signed [15:0] approx
);
    wire signed [15:0] prod;
    assign prod = x * y;
    assign exact = prod;
    assign approx = (DROP == 0) ? prod : (prod & (~((1 << DROP) - 1)));
endmodule
