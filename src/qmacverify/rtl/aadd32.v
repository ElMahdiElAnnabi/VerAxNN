module aadd32 #(parameter DROP = 0)(
    input signed [31:0] x,
    input signed [31:0] y,
    output signed [31:0] exact,
    output signed [31:0] approx
);
    wire signed [31:0] sum;
    assign sum = x + y;
    assign exact = sum;
    assign approx = (DROP == 0) ? sum : (sum & (~((1 << DROP) - 1)));
endmodule
