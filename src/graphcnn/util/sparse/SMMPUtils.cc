//We lift code from Scipy Sparse to implement some of these operations, so we
//include the below notice
/*
Copyright © 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright © 2003-2013 SciPy Developers.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

-Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

-Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

-Neither the name of Enthought nor the names of the SciPy Developers may be used to
endorse or promote products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "SMMPUtils.h"
#include <iostream>
#include <algorithm>
#include <utility>
#include <memory>

void SMMPUtils::SparseToCSRSparse(const Tensor& indices, Tensor* Ai, Tensor* Aj){
        // Initialization
    auto indices_tensor = indices.matrix<int64>();
    auto Ai_out = Ai->flat<int64>();
    auto Aj_out = Aj->flat<int64>();
    for (int64 i=0; i<Ai_out.size(); i++){
        Ai_out(i) = 0;
    }
    for (int64 i=0; i<Aj_out.size(); i++){
        Aj_out(i) = 0;
    }
    int64 activeRow = 0;
    int64 rowDiff = 0;
    for (int i = 0; i < indices.dim_size(0); i++)
    {
        rowDiff = indices_tensor(i,0) - activeRow;
        for (int i =0; i < rowDiff; i++)
        {
            activeRow++;
            Ai_out(activeRow + 1) = Ai_out(activeRow);
        }
        activeRow = indices_tensor(i,0);
        Ai_out(activeRow + 1) = i + 1;
        Aj_out(i) = indices_tensor(i,1);
    }
    //In the case where there are no indices in the last row(s), fill the rest of the Ai vector with
    //the final value
    for(int i = activeRow + 2; i < Ai_out.size(); i++)
    {
        Ai_out(i) = Ai_out(activeRow + 1);
    }
}

void SMMPUtils::CSRSparseToSparse(const Tensor& Ai, const Tensor& Aj, Tensor * indices){
    // Initialization
    auto indices_tensor = indices->matrix<int64>();
    auto indices_tensor_flat = indices->flat<int64>();
    auto Ai_out = Ai.flat<int64>();
    auto Aj_out = Aj.flat<int64>();
    for (int64 i=0; i<indices_tensor_flat.size(); i++){
        indices_tensor_flat(i) = 0;
    }
    int64 activeRow = 0;
    int64 activeRowEntryCounter = Ai_out(activeRow);
    for (int64 i=0; i < Aj_out.size(); i++)
    {
        while (activeRowEntryCounter == 0)
        {
            activeRow++;
            activeRowEntryCounter = Ai_out(activeRow) - Ai_out(activeRow - 1);
        }
        indices_tensor(i,0) = activeRow - 1;
        activeRowEntryCounter--;
        indices_tensor(i,1) = Aj_out(i);
    }
}
//Taken from Scipy's csr.h
void SMMPUtils::CsrNonzeroRows(const int64 n_row, const int64 n_col,
                    const Tensor& Ap_t,
                    const Tensor& Aj_t,
                    const Tensor& Bp_t,
                    const Tensor& Bj_t,
                    Tensor *Cp_t){

    auto Ap = Ap_t.flat<int64>();
    auto Aj = Aj_t.flat<int64>();
    auto Bp = Bp_t.flat<int64>();
    auto Bj = Bj_t.flat<int64>();
    auto Cp = Cp_t->flat<int64>();

    std::unique_ptr<std::vector<int64> > mask_ptr(new std::vector<int64>(n_col, -1));
    std::vector<int64> * mask = mask_ptr.get();
    Cp(0) = 0;

    int64 nnz = 0;
    for(size_t i = 0; i < n_row; i++){
        int64 row_nnz = 0;
        for(size_t jj = Ap(i); jj < Ap(i+1); jj++){
            int64 j = Aj(jj);
            for(size_t kk = Bp(j); kk < Bp(j+1); kk++){
                int64 k = Bj(kk);
                if((*mask)[k] != i){
                    (*mask)[k] = i;
                    row_nnz++;
                }
            }
        }
        int64 next_nnz = nnz + row_nnz;
        //std::cout << "Next NNZ or Cp(i + 1) " << next_nnz << std::endl;
        // TODO figure out equivalent exception in tensorflow
        //if (row_nnz > NPY_MAX_INTP - nnz || next_nnz != (I)next_nnz) {

        //     * Index overflowed. Note that row_nnz <= n_col and cannot overflow

        //     throw std::overflow_error("nnz of the result is too large");
        //}
        nnz = next_nnz;
        Cp(i+1) = nnz;
    }

}

//Sinister: This does not output the Cj and Cx in sorted order, but it does output the correct values
//Need to do a rowwise sort
//Taken from Scipy's csr.h
void SMMPUtils::CsrMatmul(const int64 n_row, const int64 n_cols,
               const Tensor& Ap_t,
               const Tensor& Aj_t,
               const Tensor& Ax_t,
               const Tensor& Bp_t,
               const Tensor& Bj_t,
               const Tensor& Bx_t,
               Tensor* Cp_t,
               Tensor* Cj_t,
               Tensor* Cx_t){
    auto Ap = Ap_t.flat<int64>();
    auto Aj = Aj_t.flat<int64>();
    auto Ax = Ax_t.flat<float>();
    auto Bp = Bp_t.flat<int64>();
    auto Bj = Bj_t.flat<int64>();
    auto Bx = Bx_t.flat<float>();
    auto Cp = Cp_t->flat<int64>();
    auto Cj = Cj_t->flat<int64>();
    auto Cx = Cx_t->flat<float>();
    //std::cout << n_row << " " << n_cols << std::endl;
    std::unique_ptr<std::vector<int64> > next_ptr(new std::vector<int64>(n_cols, -1));
    std::unique_ptr<std::vector<float> > sums_ptr(new std::vector<float>(n_cols, 0));
    std::vector<int64> * next = next_ptr.get();
    std::vector<float> * sums = sums_ptr.get();
    /*for (size_t i = 0; i < Ap.size(); i++)
    {
        std::cout << Ap(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i = 0; i < Ap.size(); i++)
    {
        std::cout << Ap(i) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "AX" << std::endl;
    for (size_t i =0 ; i < Ax.size(); i++)
    {
        std::cout << Ax(i) << std::endl;
    }
    std::cout << std::endl;*/

    int64 nnz = 0;
    Cp(0) = 0;
    for(size_t i = 0; i < n_row; i++){
        int64 head   = -2;
        int64 length =  0;
        int64 jj_start = Ap(i);
        int64 jj_end   = Ap(i+1);
        for(size_t jj = jj_start; jj < jj_end; jj++){
            int64 j = Aj(jj);
            //std:: cout << "J: " << j << std::endl;
            float v = Ax(jj);
            int64 kk_start = Bp(j);
            int64 kk_end   = Bp(j+1);
            for(size_t kk = kk_start; kk < kk_end; kk++){
                //std::cout << "KK: " << kk << std::endl;
                int64 k = Bj(kk);
                (*sums)[k]  += v*Bx(kk);
                if((*next)[k]  == -1){
                    (*next)[k] = head;
                    head  = k;
                    length++;
                }
            }
        }
        for(size_t jj = 0; jj < length; jj++){
            if((*sums)[head] != 0){
                Cj(nnz) = head;
                Cx(nnz) = (*sums)[head];
                nnz++;
            }
            int64 temp = head;
            head = (*next)[head];
            (*next)[temp] = -1; //clear arrays
            (*sums)[temp] =  0;
        }
        Cp(i+1) = nnz;
        //std::cout << head << " " << length << " " << jj_start << " " << jj_end << " " << i << " " << Ap(Ap.size() -1) << std::endl;
    }
    /*
    for (size_t i =0 ; i < Cp.size(); i++)
    {
        std::cout << Cp(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i =0 ; i < Cj.size(); i++)
    {
        std::cout << Cj(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i =0 ; i < Cx.size(); i++)
    {
        std::cout << Cx(i) << std::endl;
    }
    std::cout << std::endl;*/
}

//This is a variation on SMMP that puts the emphasis on only calculating for the indices Cp/Cj say they want,
//not calculating them based on Ap/Aj/Bp/Bj suggest should be the case.

//Okay this bookkeeping is confusing. To keep things straight I'm going to have the following notation:

//pPtr: the current location in the p array
//pVal: the current value in the p array
//jPtr: The current location in the j array
//jVal: The current location in the j array
//xPtr: The current location in the x array
//xVal: The current location in the x array
void SMMPUtils::CsrMatmulGradA(const int64 n_row, const int64 n_cols,
               const Tensor& Gp_t,
               const Tensor& Gj_t,
               const Tensor& Gx_t,
               const Tensor& Bp_t,
               const Tensor& Bj_t,
               const Tensor& Bx_t,
               const Tensor& dAp_t,
               const Tensor& dAj_t,
               const bool resetOutput,
               Tensor* dAx_t){
    auto Gp = Gp_t.flat<int64>();
    auto Gj = Gj_t.flat<int64>();
    auto Gx = Gx_t.flat<float>();
    auto Bp = Bp_t.flat<int64>();
    auto Bj = Bj_t.flat<int64>();
    auto Bx = Bx_t.flat<float>();
    auto dAp = dAp_t.flat<int64>();
    auto dAj = dAj_t.flat<int64>();
    auto dAx = dAx_t->flat<float>();

    /*for (size_t i = 0; i < Gp.size(); i++)
    {
        std::cout << Gp(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i = 0; i < Gj.size(); i++)
    {
        std::cout << Gj(i) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "PT" << std::endl;
    for (size_t i =0 ; i < Gx.size(); i++)
    {
        std::cout << Gx(i) << std::endl;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < Bp.size(); i++)
    {
        std::cout << Bp(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i = 0; i < Bj.size(); i++)
    {
        std::cout << Bj(i) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "G" << std::endl;
    for (size_t i =0 ; i < Bx.size(); i++)
    {
        std::cout << Bx(i) << std::endl;
    }
    std::cout << std::endl;

        for (size_t i = 0; i < dAp.size(); i++)
    {
        std::cout << dAp(i) << std::endl;
    }
    std::cout << std::endl;
    for (size_t i = 0; i < dAj.size(); i++)
    {
        std::cout << dAj(i) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "DAX" << std::endl;
    for (size_t i =0 ; i < dAx.size(); i++)
    {
        std::cout << dAx(i) << std::endl;
    }
    std::cout << std::endl;*/

    //int64 nnz = 0;

    //Get output row
    for(size_t dApPtr = 0; dApPtr < n_row; dApPtr++)
    {
        //std::cout << "dB I Ptr (the row for dB): " << dApPtr << std::endl;
        //std::cout << "dB I val : " << dAp(dApPtr) << std::endl;
        //Count columns per row
        int64 dAjPtr_start = dAp(dApPtr);
        int64 dAjPtr_end = dAp(dApPtr+1);
        for(size_t dAjPtr = dAjPtr_start; dAjPtr < dAjPtr_end; dAjPtr++)
        {
            //std::cout << "dB J Ptr: " << dAjPtr << std::endl;
            //std::cout << "dB J val (the column for dB): " << dAj(dAjPtr) << std::endl;
            int64 GiPtr = dApPtr;
            //std::cout << "AT i ptr (the row for AT)" << GiPtr << std::endl;
            //std::cout << "AT i val   " << Gp(GiPtr) << std::endl;
            int64 dAjVal = dAj(dAjPtr);
            int64 Gjptr_start = Gp(GiPtr);
            int64 Gjptr_end = Gp(GiPtr+1);
            int64 BiPtr = dAjVal;
            //std::cout << "G i ptr (the row for G)" << BiPtr << std::endl;
            //std::cout << "G i val   " << Bp(BiPtr) << std::endl;
            int64 Bjptr_start = Bp(BiPtr);
            int64 Bjptr_end = Bp(BiPtr+1);
            int64 BjPtr = Bjptr_start;
            int64 BjVal = Bj(BjPtr);
            //std::cout << "G j ptr " << BjPtr << std::endl;
            //std::cout << "G j val (the column for G) " << BjVal << std::endl;
            //There's a vicious subtlety here: In the case of multiplying a single layer matrix by a stack of
            //matrices, you do NOT want to reset the output after the first go around because you want the values
            //to accumulate in that single-layer matrix (because the derivative should be summed across all layers.
            //Make sure this boolean is set to get the behavior you want!
            if (resetOutput)
            {
                dAx(dAjPtr) = 0;
            }
            for (size_t Gjptr = Gjptr_start; Gjptr < Gjptr_end; Gjptr++)
            {
                int64 GjVal = Gj(Gjptr);
                //std::cout << "AT j ptr " << Gjptr << std::endl;
                //std::cout << "AT j val (the column for AT) " << GjVal << std::endl;
                if (BjPtr < Bjptr_end)
                {
                    //else if (BjVal > GjVal)
                    //{
                    //    break;
                    //}
                    while (BjVal < GjVal)
                    {
                        BjPtr++;
                        if (BjPtr >= Bjptr_end)
                        {
                            break;
                        }
                        BjVal = Bj(BjPtr);
                        //std::cout << "G j val (the column for G) " << BjVal << std::endl;
                    }
                    if (BjVal == GjVal)
                    {
                        dAx(dAjPtr) += Gx(Gjptr)*Bx(BjPtr);
                        //std::cout << "Multiply " << BjVal << " with " << GjVal << std::endl;
                        BjPtr++;
                        BjVal = Bj(BjPtr);
                        //std::cout << "G j val (the column for G) " << BjVal << std::endl;
                        //break;
                    }
                }
                else
                {
                    break;
                }

            }
            /*for (size_t Bjptr = Bjptr_start; Bjptr < Bjptr_end; Bjptr++)
            {
                int64 BjPtr = Bjptr;
                int64 BjVal = Bj(BjVal);
                std::cout << "B j ptr " << Bjptr << std::endl;
                std::cout << "B j val (the column for B) " << Bj(BjPtr) << std::endl;
            }*/
        }

    }
}

//Taken from Scipy's csr.h
void SMMPUtils::CsrTranspose(const int64 n_row, const int64 n_col,
               const Tensor& Ap_t,
               const Tensor& Aj_t,
               const Tensor& Ax_t,
               Tensor* Bp_t,
               Tensor* Bi_t,
               Tensor* Bx_t)
{
    auto Ap = Ap_t.flat<int64>();
    auto Aj = Aj_t.flat<int64>();
    auto Ax = Ax_t.flat<float>();
    auto Bp = Bp_t->flat<int64>();
    auto Bi = Bi_t->flat<int64>();
    auto Bx = Bx_t->flat<float>();

    const int64 nnz = Ap(n_row);

    //compute number of non-zero entries per column of A
    for (size_t n = 0; n < n_col; n++)
    {
        Bp(n) = 0;
    }

    for (size_t n = 0; n < nnz; n++){
        Bp(Aj(n))++;
    }

    //cumsum the nnz per column to get Bp[]
    for(size_t col = 0, cumsum = 0; col < n_col; col++){
        int64 temp  = Bp(col);
        Bp(col) = cumsum;
        cumsum += temp;
    }
    Bp(n_col) = nnz;
    //std::cout << "LOL2" << std::endl;

    for(size_t row = 0; row < n_row; row++){
        for(size_t jj = Ap(row); jj < Ap(row+1); jj++){
            int64 col  = Aj(jj);
            int64 dest = Bp(col);
            //std::cout << row << " " << jj << " " << col << " " << dest << " " A
            Bi(dest) = row;
            Bx(dest) = Ax(jj);

            Bp(col)++;
        }
    }

    for(size_t col = 0, last = 0; col <= n_col; col++){
        int64 temp  = Bp(col);
        Bp(col) = last;
        last    = temp;
    }
}
void SMMPUtils::SortIndices(const Tensor& Ai_t, Tensor* Aj_t)
{
    auto Ai = Ai_t.flat<int64>();
    auto Aj = Aj_t->flat<int64>();
    std::vector<int64> buffer(Aj.size(),0);

    for (size_t i = 0; i < Aj.size(); i++)
    {
        buffer[i] = Aj(i);
    }
    for (size_t i = 1; i < Ai.size(); i++)
    {
        std::sort(buffer.begin() + Ai(i - 1),buffer.begin() + Ai(i));
    }
    for (size_t i = 0; i < Aj.size(); i++)
    {
        Aj(i) = buffer[i];
        //std::cout << Aj(i) << std::endl;
    }
    //std::cout << std::endl;
}